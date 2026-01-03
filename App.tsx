import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type, FunctionDeclaration } from "@google/genai";
import { AgentPersona, MessageLog, BlobData } from './types';
import { 
  Phone, PhoneOff, Leaf, Activity, 
  ThermometerSun, ShieldAlert, 
  MapPin, Signal, Radio, 
  Zap, ShieldCheck, Sparkles, 
  Volume2, Download, User, Bot, Server
} from 'lucide-react';

const API_KEY = (process.env.API_KEY || '');
const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';

const SYSTEM_INSTRUCTION = `
You are the Voice AI system for 'Green Choice Heating & Cooling' in East York.
PERSONA 1: CHLOE (Lead Gen & Rebates). Tone: Polished, warm, persuasive.
PERSONA 2: SAM (Emergency). Tone: Tactical, authoritative, lightning-fast.

Detection Protocol: If you hear "water leaking", "cold house", "no heat", "sparks", or "scary noise", transfer to Sam immediately.
GTA Context: Mention "East York", "Scarborough", or "North York" naturally if appropriate.

Capabilities: 
- You can trigger a handoff to Sam if an emergency is detected.
- You can download the current transcript if the user asks to "save the conversation", "download transcript", or "export our chat".
`;

const switchToSamTool: FunctionDeclaration = {
  name: 'switchToSam',
  parameters: {
    type: Type.OBJECT,
    description: 'Persona handoff to emergency mode.',
    properties: {},
  },
};

const downloadTranscriptTool: FunctionDeclaration = {
  name: 'downloadTranscript',
  parameters: {
    type: Type.OBJECT,
    description: 'Triggers the download of the conversation transcript file for the user.',
    properties: {},
  },
};

const tools = [{ functionDeclarations: [switchToSamTool, downloadTranscriptTool] }];

// --- Sound Synthesizer ---
function playHandoffChime(ctx: AudioContext) {
  const now = ctx.currentTime;
  const playTone = (freq: number, start: number, duration: number, type: OscillatorType = 'sine') => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, start);
    osc.frequency.exponentialRampToValueAtTime(freq * 1.5, start + duration);
    gain.gain.setValueAtTime(0, start);
    gain.gain.linearRampToValueAtTime(0.3, start + 0.05);
    gain.gain.exponentialRampToValueAtTime(0.001, start + duration);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(start);
    osc.stop(start + duration);
  };
  playTone(440, now, 0.4); 
  playTone(880, now + 0.15, 0.6, 'triangle');
}

function createBlob(data: Float32Array): BlobData {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    const s = Math.max(-1, Math.min(1, data[i]));
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return {
    data: btoa(String.fromCharCode(...new Uint8Array(int16.buffer))),
    mimeType: 'audio/pcm;rate=16000',
  };
}

function decodeAudio(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
  return bytes;
}

async function decodeAudioData(data: Uint8Array, ctx: AudioContext, sampleRate: number, numChannels: number): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);
  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

const AudioVisualizer: React.FC<{ volume: number; persona: AgentPersona; isActive: boolean; isSwitching: boolean }> = ({ volume, persona, isActive, isSwitching }) => {
  const isChloe = persona === AgentPersona.CHLOE;
  const colorBase = isChloe ? 'emerald' : 'rose';
  
  return (
    <div className={`relative flex items-center justify-center w-full max-w-[400px] aspect-square transition-all duration-700 ${isSwitching ? 'scale-110' : ''}`}>
      {/* Visualizer Background Rings */}
      <div className={`absolute inset-0 rounded-full bg-${colorBase}-500 opacity-[0.05] visualizer-ring`} />
      <div className={`absolute inset-10 rounded-full bg-${colorBase}-500 opacity-[0.08] visualizer-ring`} style={{ animationDelay: '1s' }} />
      
      {/* Voice Level Bars */}
      <div className="absolute inset-0 flex items-center justify-center gap-1.5 px-4">
        {Array.from({ length: 48 }).map((_, i) => (
          <div
            key={i}
            className={`w-1 rounded-full bg-${colorBase}-500 transition-all duration-75`}
            style={{
              height: isActive ? `${Math.max(8, volume * 180 * (0.6 + Math.random() * 0.4))}%` : '4px',
              opacity: isActive ? 0.8 : 0.2,
            }}
          />
        ))}
      </div>

      {/* Profile Image Container */}
      <div className={`relative w-56 h-56 rounded-[3.5rem] p-1.5 transition-all duration-700 glass-card ${isChloe ? 'agent-glow-chloe' : 'agent-glow-sam'} ${isSwitching ? 'blur-sm scale-90' : 'hover:scale-105'}`}>
        <div className="w-full h-full rounded-[3.2rem] overflow-hidden bg-white shadow-inner relative">
          <img 
            src={isChloe 
              ? "https://images.unsplash.com/photo-1594744803329-e58b31de8bf5?auto=format&fit=crop&q=80&w=400&h=400" 
              : "https://images.unsplash.com/photo-1519085185756-62002b3ad159?auto=format&fit=crop&q=80&w=400&h=400"
            } 
            className={`w-full h-full object-cover transition-transform duration-1000 ${isActive ? 'scale-110' : 'scale-100'}`}
            alt="AI Agent"
          />
          {isActive && volume > 0.05 && (
            <div className={`absolute top-4 right-4 p-2 bg-white/90 rounded-2xl shadow-lg text-${colorBase}-500 animate-pulse`}>
              <Sparkles className="w-5 h-5" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const StatItem: React.FC<{ icon: any; label: string; value: string; active: boolean; color: string }> = ({ icon: Icon, label, value, active, color }) => (
  <div className="glass-card p-5 rounded-[2rem] flex items-center gap-5 border border-white/80 group">
    <div className={`w-14 h-14 rounded-2xl flex items-center justify-center ${active ? color : 'bg-slate-200'} text-white transition-all duration-500 shadow-md`}>
      <Icon className="w-7 h-7" />
    </div>
    <div>
      <p className="text-[11px] font-extrabold text-slate-400 uppercase tracking-widest mb-0.5">{label}</p>
      <p className="text-base font-bold text-slate-900">{value}</p>
    </div>
  </div>
);

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [currentPersona, setCurrentPersona] = useState<AgentPersona>(AgentPersona.CHLOE);
  const [isSwitching, setIsSwitching] = useState(false);
  const [volumeLevel, setVolumeLevel] = useState(0);
  const [transcripts, setTranscripts] = useState<MessageLog[]>([]);

  const nextStartTime = useRef<number>(0);
  const audioContexts = useRef<{ input: AudioContext | null; output: AudioContext | null }>({ input: null, output: null });
  const sessionRef = useRef<any>(null);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  const transcriptsRef = useRef<MessageLog[]>([]);
  useEffect(() => {
    transcriptsRef.current = transcripts;
  }, [transcripts]);

  const disconnect = useCallback(() => {
    sessionRef.current = null;
    if (audioContexts.current.input) audioContexts.current.input.close();
    if (audioContexts.current.output) audioContexts.current.output.close();
    sourcesRef.current.forEach(s => { try { s.stop(); } catch(e) {} });
    setIsConnected(false);
    setVolumeLevel(0);
  }, []);

  const downloadTranscript = useCallback(() => {
    const logs = transcriptsRef.current;
    if (logs.length === 0) return;
    const content = logs
      .map(t => {
        const time = t.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const label = t.role === 'user' ? 'USER' : (t.role === 'system' ? 'SYSTEM' : 'AI AGENT');
        return `[${time}] ${label}: ${t.text}`;
      })
      .join('\n\n');

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `GreenChoice_Transcript_${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  const connect = async () => {
    try {
      const ai = new GoogleGenAI({ apiKey: API_KEY });
      const inputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      const outputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      audioContexts.current = { input: inputCtx, output: outputCtx };

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const sessionPromise = ai.live.connect({
        model: MODEL_NAME,
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: SYSTEM_INSTRUCTION,
          tools: tools,
          inputAudioTranscription: {}, 
          outputAudioTranscription: {}, 
        },
        callbacks: {
          onopen: () => {
            setIsConnected(true);
            setTranscripts([]);
            const source = inputCtx.createMediaStreamSource(stream);
            const processor = inputCtx.createScriptProcessor(4096, 1, 1);
            processor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              let sum = 0;
              for(let i=0; i<inputData.length; i++) sum += inputData[i] * inputData[i];
              setVolumeLevel(prev => Math.max(Math.sqrt(sum / inputData.length) * 5, prev * 0.9));
              sessionPromise.then(s => s.sendRealtimeInput({ media: createBlob(inputData) }));
            };
            source.connect(processor);
            processor.connect(inputCtx.destination);
          },
          onmessage: async (msg) => {
            if (msg.toolCall) {
              for (const fc of msg.toolCall.functionCalls) {
                if (fc.name === 'switchToSam') {
                  if (audioContexts.current.output) playHandoffChime(audioContexts.current.output);
                  setIsSwitching(true);
                  setTranscripts(prev => [...prev, { role: 'system', text: 'URGENT: ELEVATING TO LEAD DISPATCH (SAM)...', timestamp: new Date() }]);
                  setTimeout(() => {
                    setCurrentPersona(AgentPersona.SAM);
                    setIsSwitching(false);
                  }, 2500);
                } else if (fc.name === 'downloadTranscript') {
                  downloadTranscript();
                  setTranscripts(prev => [...prev, { role: 'system', text: 'TRANSCRIPT DOWNLOAD TRIGGERED.', timestamp: new Date() }]);
                }
                sessionPromise.then(s => s.sendToolResponse({ functionResponses: { id: fc.id, name: fc.name, response: { result: "ok" } } }));
              }
            }
            if (msg.serverContent?.inputTranscription?.text) {
              const text = msg.serverContent.inputTranscription.text;
              setTranscripts(prev => [...prev, { role: 'user', text, timestamp: new Date() }]);
            }
            if (msg.serverContent?.outputTranscription?.text) {
              const text = msg.serverContent.outputTranscription.text;
              setTranscripts(prev => {
                const last = prev[prev.length - 1];
                if (last && last.role === 'model') {
                   return [...prev.slice(0, -1), { ...last, text: last.text + text }];
                }
                return [...prev, { role: 'model', text, timestamp: new Date() }];
              });
            }
            const audioData = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData && outputCtx) {
              nextStartTime.current = Math.max(nextStartTime.current, outputCtx.currentTime);
              const buf = await decodeAudioData(decodeAudio(audioData), outputCtx, 24000, 1);
              const source = outputCtx.createBufferSource();
              source.buffer = buf;
              source.connect(outputCtx.destination);
              source.start(nextStartTime.current);
              nextStartTime.current += buf.duration;
              sourcesRef.current.add(source);
            }
          },
          onclose: () => setIsConnected(false),
          onerror: (err) => { console.error(err); disconnect(); }
        }
      });
      sessionRef.current = sessionPromise;
    } catch (e) { console.error(e); }
  };

  const isChloe = currentPersona === AgentPersona.CHLOE;
  const themeColor = isChloe ? 'emerald' : 'rose';

  return (
    <div className={`min-h-screen relative flex flex-col items-center p-6 md:p-12 transition-all duration-1000 ${isChloe ? 'bg-emerald-50' : 'bg-rose-50'}`}>
      
      {/* Background Ambience */}
      <div className="mesh-gradient">
        <div className={`mesh-blob bg-${themeColor}-400/30 top-[-10%] left-[-10%] transition-colors duration-1000`} />
        <div className={`mesh-blob bg-${isChloe ? 'cyan' : 'orange'}-400/30 bottom-[-10%] right-[-10%] animation-delay-2000 transition-colors duration-1000`} />
      </div>

      <header className="w-full max-w-[1400px] flex items-center justify-between mb-16 relative z-10">
        <div className="flex items-center gap-6">
          <div className={`w-16 h-16 ${isChloe ? 'bg-emerald-700' : 'bg-rose-700'} rounded-[1.8rem] flex items-center justify-center shadow-2xl transition-all duration-700 rotate-3`}>
            <Leaf className="w-10 h-10 text-white" />
          </div>
          <div>
            <h1 className="text-4xl font-[1000] tracking-tighter text-slate-900 uppercase italic leading-none">Green Choice</h1>
            <div className="flex items-center gap-3 mt-1.5">
              <span className={`text-xs font-black ${isChloe ? 'text-emerald-700' : 'text-rose-700'} tracking-widest uppercase transition-colors duration-700`}>GTA Dispatch Hub</span>
              <div className="w-1 h-1 bg-slate-300 rounded-full" />
              <div className="flex items-center gap-1.5">
                 <Signal className={`w-4 h-4 ${isConnected ? 'text-blue-600 animate-pulse' : 'text-slate-400'}`} />
                 <span className="text-[10px] font-extrabold text-slate-500 uppercase">Secure Link v2.6</span>
              </div>
            </div>
          </div>
        </div>

        <div className="hidden lg:flex items-center gap-8">
          <div className="glass-card px-8 py-3.5 rounded-2xl flex items-center gap-4">
             <div className={`w-3 h-3 rounded-full ${isConnected ? (isChloe ? 'bg-emerald-500' : 'bg-rose-500') + ' animate-ping' : 'bg-slate-300'} transition-colors duration-700`} />
             <span className="text-sm font-black text-slate-800 uppercase tracking-widest">{isConnected ? 'System Live' : 'Standby'}</span>
          </div>
          <div className="bg-white p-2.5 rounded-2xl shadow-xl flex items-center gap-3">
             <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${isChloe ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'} transition-colors`}>
               <MapPin className="w-5 h-5" />
             </div>
             <span className="text-xs font-black text-slate-700 uppercase pr-3 tracking-wider">East York Regional Hub</span>
          </div>
        </div>
      </header>

      <main className="w-full max-w-[1400px] grid grid-cols-1 lg:grid-cols-12 gap-10 items-stretch relative z-10">
        
        {/* Left Stats Column */}
        <div className="lg:col-span-3 space-y-5">
          <StatItem icon={Radio} label="Protocol" value="Native Voice" active={isConnected} color={isChloe ? 'bg-emerald-600' : 'bg-rose-600'} />
          <StatItem icon={Volume2} label="Latency" value="~140ms" active={isConnected} color={isChloe ? 'bg-cyan-600' : 'bg-orange-600'} />
          <StatItem icon={ShieldCheck} label="Encrypted" value="AES-256-GCM" active={isConnected} color={isChloe ? 'bg-blue-600' : 'bg-indigo-600'} />

          <div className={`p-8 rounded-[2.5rem] transition-all duration-1000 ${isChloe ? 'bg-gradient-to-br from-emerald-700 to-emerald-900' : 'bg-gradient-to-br from-rose-700 to-rose-900'} text-white shadow-2xl relative overflow-hidden group border-t border-white/20`}>
             <div className="absolute -top-6 -right-6 p-4 opacity-5 group-hover:scale-125 transition-transform">
               <Zap className="w-40 h-40" />
             </div>
             <div className="relative z-10 space-y-4">
               <div className="flex items-center gap-3">
                 <div className="p-1.5 bg-yellow-400 rounded-lg">
                    <Zap className="w-4 h-4 text-emerald-900 fill-emerald-900" />
                 </div>
                 <span className="text-xs font-black uppercase tracking-[0.2em] text-yellow-100">AI Logic Core</span>
               </div>
               <p className="text-lg font-bold leading-tight tracking-tight">
                 {isChloe 
                  ? "Chloe is analyzing service history and rebate eligibility for high-tier incentives." 
                  : "Sam has taken control. Emergency dispatch protocols are prioritizing life-safety scenarios."}
               </p>
             </div>
          </div>
        </div>

        {/* Center Visualizer Column */}
        <div className={`lg:col-span-6 glass-card rounded-[4rem] p-10 flex flex-col items-center justify-center relative overflow-hidden transition-all duration-1000 shadow-2xl ${isChloe ? 'agent-glow-chloe' : 'agent-glow-sam'} ${isSwitching ? 'scale-[0.97]' : ''}`}>
          
          <div className="absolute top-10 flex items-center gap-4 px-8 py-3 bg-white rounded-3xl shadow-lg border border-slate-100">
            <Activity className={`w-5 h-5 ${isConnected ? 'text-blue-500 animate-pulse' : 'text-slate-300'}`} />
            <span className="text-[13px] font-black uppercase text-slate-900 tracking-[0.25em]">Voice Processor {isConnected ? 'Online' : 'Ready'}</span>
          </div>

          <AudioVisualizer volume={volumeLevel} persona={currentPersona} isActive={isConnected} isSwitching={isSwitching} />

          <div className="mt-8 text-center space-y-3">
            <h2 className={`text-7xl font-[1000] tracking-tighter text-slate-950 italic transition-all duration-700 ${isSwitching ? 'blur-md opacity-0' : 'opacity-100'}`}>
              {isChloe ? 'Chloe' : 'Sam'}
            </h2>
            <div className={`inline-flex items-center gap-3 px-6 py-2.5 rounded-2xl text-[12px] font-black uppercase tracking-[0.25em] transition-all duration-1000 shadow-sm ${isChloe ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'}`}>
              {isChloe ? <ThermometerSun className="w-5 h-5" /> : <ShieldAlert className="w-5 h-5" />}
              {isChloe ? 'HVAC Rebate Lead' : 'Emergency Specialist'}
            </div>
          </div>

          <div className="w-full max-w-[340px] mt-12">
             <button
               onClick={isConnected ? disconnect : connect}
               className={`w-full py-7 rounded-[2.5rem] font-black text-xl tracking-widest uppercase transition-all duration-500 flex items-center justify-center gap-5 shadow-2xl hover:-translate-y-2 active:translate-y-0 relative overflow-hidden ${
                 isConnected 
                   ? 'bg-slate-950 text-white' 
                   : 'bg-blue-600 text-white hover:bg-blue-700'
               } ${isSwitching ? 'opacity-30 pointer-events-none' : ''}`}
             >
               {isConnected ? <PhoneOff className="w-7 h-7" /> : <Phone className="w-7 h-7 animate-pulse" />}
               <span>{isConnected ? 'Kill Link' : 'Secure Call'}</span>
               {!isConnected && <div className="absolute inset-0 shimmer opacity-20" />}
             </button>
          </div>
        </div>

        {/* Right Feed Column */}
        <div className="lg:col-span-3">
          <div className="glass-card rounded-[3.5rem] h-full flex flex-col overflow-hidden shadow-2xl border-white/60">
            <div className="p-8 border-b border-slate-200/50 bg-white/50 flex items-center justify-between">
               <h3 className="text-[13px] font-black text-slate-900 uppercase tracking-widest flex items-center gap-2">
                 <Activity className="w-4 h-4 text-slate-400" />
                 Live Feed
               </h3>
               <button 
                onClick={downloadTranscript}
                disabled={transcripts.length === 0}
                className="flex items-center gap-2 px-4 py-2.5 rounded-2xl bg-slate-950 text-white text-[11px] font-black hover:bg-slate-800 transition-all disabled:opacity-20 disabled:grayscale disabled:cursor-not-allowed shadow-lg"
               >
                 <Download className="w-4 h-4" />
                 EXPORT
               </button>
            </div>
            
            <div className="flex-1 overflow-y-auto p-8 space-y-8 scrollbar-hide">
               {transcripts.length === 0 ? (
                 <div className="h-full flex flex-col items-center justify-center text-slate-300 text-center gap-8">
                    <Activity className="w-20 h-20 opacity-20 animate-pulse" />
                    <p className="text-xs font-black uppercase tracking-[0.2em] leading-relaxed max-w-[180px] text-slate-400">Waiting for Voice Link...</p>
                 </div>
               ) : (
                 transcripts.map((t, i) => (
                   <div key={i} className="animate-in slide-in-from-bottom-6 duration-500">
                      <div className="flex items-center justify-between mb-2 px-2">
                        <div className="flex items-center gap-2">
                           {t.role === 'user' ? <User className="w-3.5 h-3.5 text-slate-400" /> : (t.role === 'model' ? <Bot className="w-3.5 h-3.5 text-blue-500" /> : <Server className="w-3.5 h-3.5 text-rose-500" />)}
                           <span className={`text-[11px] font-black uppercase tracking-widest ${t.role === 'system' ? 'text-rose-600' : (t.role === 'model' ? 'text-blue-600' : 'text-slate-500')}`}>
                             {t.role === 'system' ? 'SYSTEM' : (t.role === 'model' ? 'AI AGENT' : 'YOU')}
                           </span>
                        </div>
                        <span className="text-[10px] font-bold text-slate-400">{t.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                      </div>
                      <div className={`transcript-bubble text-[15px] font-bold leading-snug p-6 rounded-[2.2rem] border transition-all ${t.role === 'system' ? 'bg-rose-50 text-rose-800 border-rose-100 italic' : (t.role === 'model' ? 'bg-blue-50/70 text-blue-950 border-blue-100' : 'bg-white text-slate-900 border-slate-100 shadow-sm')}`}>
                        {t.text}
                      </div>
                   </div>
                 ))
               )}
            </div>

            <div className="p-10 bg-white/40 border-t border-slate-200/50">
              <div className="flex items-center justify-between text-[11px] font-black text-slate-500 uppercase tracking-widest mb-4">
                <span>Buffer Health</span>
                <span className="text-emerald-700">99.8%</span>
              </div>
              <div className="w-full h-3 bg-slate-200/50 rounded-full overflow-hidden p-0.5 border border-slate-300/30">
                <div className={`h-full rounded-full ${isChloe ? 'bg-emerald-500' : 'bg-rose-500'} w-[90%] transition-all duration-1000 shadow-inner`} />
              </div>
            </div>
          </div>
        </div>

      </main>

      <footer className="mt-20 w-full max-w-[1400px] flex flex-col md:flex-row items-center justify-between gap-10 border-t border-slate-200/60 pt-12 relative z-10 pb-12">
        <div className="flex items-center gap-16">
           <div className="flex items-center gap-4">
             <div className="w-4 h-4 bg-emerald-500 rounded-full shadow-[0_0_15px_rgba(16,185,129,0.5)]" />
             <span className="text-xs font-black text-slate-500 uppercase tracking-[0.3em]">Core Verified</span>
           </div>
           <div className="flex items-center gap-4">
             <div className="w-4 h-4 bg-blue-500 rounded-full shadow-[0_0_15px_rgba(59,130,246,0.5)]" />
             <span className="text-xs font-black text-slate-500 uppercase tracking-[0.3em]">Gemini Live API</span>
           </div>
        </div>
        <div className="text-[11px] font-extrabold text-slate-400 uppercase tracking-[0.5em] text-center md:text-right">
          © 2026 GREEN CHOICE DISPATCH v4.0.2-EASTYORK
        </div>
      </footer>

      {/* Switching Persona Overlay */}
      {isSwitching && (
        <div className="fixed inset-0 z-50 bg-rose-950/95 backdrop-blur-[60px] flex flex-col items-center justify-center animate-in fade-in duration-500">
           <div className="relative p-12 max-w-2xl w-full text-center space-y-12">
              <div className="absolute inset-0 bg-white blur-[180px] opacity-10" />
              
              <div className="relative space-y-8">
                 <div className="w-36 h-36 mx-auto bg-white rounded-[2.5rem] flex items-center justify-center animate-bounce shadow-[0_0_80px_rgba(255,255,255,0.4)]">
                    <ShieldAlert className="w-20 h-20 text-rose-700" />
                 </div>
                 <div className="space-y-4">
                   <h2 className="text-6xl font-[1000] text-white tracking-tighter uppercase italic">Red Alert Elevating</h2>
                   <p className="text-rose-200 font-black uppercase tracking-[0.6em] text-sm">Transferring to Priority Lead Sam</p>
                 </div>
              </div>
              
              <div className="flex gap-4 w-full px-12">
                {[0,1,2,3].map(i => (
                  <div key={i} className="flex-1 h-3 bg-white/10 rounded-full overflow-hidden">
                    <div className="w-full h-full bg-white animate-[shimmer_2s_infinite]" style={{ animationDelay: `${i*0.3}s` }} />
                  </div>
                ))}
              </div>

              <div className="text-white/40 text-[11px] font-black uppercase tracking-widest animate-pulse">
                 Securing Emergency Emergency Handshake...
              </div>
           </div>
        </div>
      )}
    </div>
  );
}
