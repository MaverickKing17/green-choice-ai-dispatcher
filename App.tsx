import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type, FunctionDeclaration } from "@google/genai";
import { AgentPersona, MessageLog, BlobData } from './types';
import { 
  Phone, PhoneOff, Leaf, Activity, 
  ArrowRightLeft, ThermometerSun, ShieldAlert, 
  MapPin, Signal, Radio, 
  Zap, Clock, Globe, ShieldCheck, Sparkles, 
  Volume2
} from 'lucide-react';

const API_KEY = (process.env.API_KEY || '');
const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';

const SYSTEM_INSTRUCTION = `
You are the Voice AI system for 'Green Choice Heating & Cooling' in East York.
PERSONA 1: CHLOE (Lead Gen & Rebates). Tone: Polished, warm, persuasive.
PERSONA 2: SAM (Emergency). Tone: Tactical, authoritative, lightning-fast.

Detection Protocol: If you hear "water leaking", "cold house", "no heat", "sparks", or "scary noise", transfer to Sam immediately.
GTA Context: Mention "East York", "Scarborough", or "North York" naturally if appropriate.
`;

const switchToSamTool: FunctionDeclaration = {
  name: 'switchToSam',
  description: 'Persona handoff to emergency mode.',
  parameters: { type: Type.OBJECT, properties: {} },
};

const startSurveyTool: FunctionDeclaration = {
  name: 'startSurvey',
  description: 'Wrap up call.',
  parameters: { type: Type.OBJECT, properties: {} },
};

const tools = [{ functionDeclarations: [switchToSamTool, startSurveyTool] }];

// --- Sound Synthesizer ---
function playHandoffChime(ctx: AudioContext) {
  const now = ctx.currentTime;
  
  // Create a pleasant but urgent 2-tone chime
  const playTone = (freq: number, start: number, duration: number, type: OscillatorType = 'sine') => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, start);
    osc.frequency.exponentialRampToValueAtTime(freq * 1.5, start + duration);
    
    gain.gain.setValueAtTime(0, start);
    gain.gain.linearRampToValueAtTime(0.2, start + 0.05);
    gain.gain.exponentialRampToValueAtTime(0.001, start + duration);
    
    osc.connect(gain);
    gain.connect(ctx.destination);
    
    osc.start(start);
    osc.stop(start + duration);
  };

  playTone(440, now, 0.4); // A4
  playTone(880, now + 0.15, 0.6, 'triangle'); // A5
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

const AudioVisualizerPortal: React.FC<{ volume: number; persona: AgentPersona; isActive: boolean; isSwitching: boolean }> = ({ volume, persona, isActive, isSwitching }) => {
  const isChloe = persona === AgentPersona.CHLOE;
  const primaryColor = isChloe ? 'from-emerald-400' : 'from-rose-500';
  const secondaryColor = isChloe ? 'to-cyan-400' : 'to-orange-400';
  const accentColor = isChloe ? 'emerald' : 'rose';
  
  const scale = 1 + (volume * 1.5);

  return (
    <div className={`relative flex items-center justify-center w-80 h-80 transition-all duration-300 ${isSwitching ? 'scale-110 rotate-6' : ''}`}>
      {/* Background Blooming Glow */}
      <div 
        className={`absolute inset-0 rounded-full bg-gradient-to-br ${primaryColor} ${secondaryColor} ${isSwitching ? 'opacity-40 blur-[100px]' : 'opacity-[0.15] blur-[60px]'} transition-all duration-700`}
        style={{ transform: `scale(${scale * 1.4})` }}
      />
      
      {/* Dynamic Spectrum Rings */}
      <div 
        className={`absolute inset-0 rounded-full border-[3px] border-${accentColor}-500/10 transition-all duration-300 ${isSwitching ? 'border-white animate-ping' : ''}`}
        style={{ transform: `scale(${scale * 1.15})` }}
      />

      {/* Radial Frequency Spectrum */}
      <div className="absolute inset-0 flex items-center justify-center">
        {Array.from({ length: 60 }).map((_, i) => (
          <div
            key={i}
            className={`absolute w-1 rounded-full bg-gradient-to-t ${primaryColor} ${secondaryColor} transition-all duration-75 ${isSwitching ? 'bg-white h-[40%]' : ''}`}
            style={{
              height: isActive ? `${Math.max(12, volume * 150 * (0.7 + Math.random() * 0.3))}%` : '4px',
              transform: `rotate(${i * 6}deg) translateY(-115px)`,
              opacity: isActive ? 1 : 0.1,
              filter: `drop-shadow(0 0 5px ${isChloe ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'})`
            }}
          />
        ))}
      </div>

      {/* The Agent Portal */}
      <div className={`relative w-48 h-48 rounded-[3rem] p-1.5 rotate-3 transition-all duration-700 ${isChloe ? 'agent-glow-chloe bg-emerald-500/10' : 'agent-glow-sam bg-rose-500/10'} overflow-hidden shadow-2xl ${isSwitching ? 'blur-sm scale-95 opacity-50' : ''}`}>
        <div className="w-full h-full rounded-[2.8rem] overflow-hidden bg-white ring-4 ring-white/80 shadow-inner">
          <img 
            src={isChloe 
              ? "https://images.unsplash.com/photo-1594744803329-e58b31de8bf5?auto=format&fit=crop&q=80&w=400&h=400" 
              : "https://images.unsplash.com/photo-1519085185756-62002b3ad159?auto=format&fit=crop&q=80&w=400&h=400"
            } 
            className={`w-full h-full object-cover transition-all duration-1000 ${isActive ? 'scale-110' : 'scale-100 grayscale-[0.5]'}`}
            alt="AI Agent"
          />
        </div>
        {isActive && volume > 0.1 && !isSwitching && (
            <Sparkles className={`absolute top-4 right-4 w-6 h-6 text-${accentColor}-400 animate-pulse`} />
        )}
      </div>

      {/* Transition Flash */}
      {isSwitching && (
        <div className="absolute inset-0 z-20 bg-white rounded-full animate-pulse opacity-20" />
      )}
    </div>
  );
};

const HighVisibilityMetric: React.FC<{ icon: any; label: string; value: string; colorClass: string }> = ({ icon: Icon, label, value, colorClass }) => (
  <div className="glass-container p-4 rounded-3xl flex items-center gap-4 group hover:bg-white/60 transition-all duration-300">
    <div className={`w-12 h-12 rounded-2xl flex items-center justify-center ${colorClass} text-white shadow-lg`}>
      <Icon className="w-6 h-6" />
    </div>
    <div className="flex flex-col">
      <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{label}</span>
      <span className="text-sm font-extrabold text-slate-800">{value}</span>
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

  const disconnect = useCallback(() => {
    sessionRef.current = null;
    if (audioContexts.current.input) audioContexts.current.input.close();
    if (audioContexts.current.output) audioContexts.current.output.close();
    sourcesRef.current.forEach(s => { try { s.stop(); } catch(e) {} });
    setIsConnected(false);
    setVolumeLevel(0);
  }, []);

  const connect = async () => {
    try {
      const ai = new GoogleGenAI({ apiKey: API_KEY });
      const inputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      const outputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      audioContexts.current = { input: inputCtx, output: outputCtx };

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const session = await ai.live.connect({
        model: MODEL_NAME,
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: SYSTEM_INSTRUCTION,
          tools: tools,
          inputAudioTranscription: { model: 'gemini-2.5-flash-native-audio-preview-09-2025' }, 
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
              session.sendRealtimeInput({ media: createBlob(inputData) });
            };
            source.connect(processor);
            processor.connect(inputCtx.destination);
          },
          onmessage: async (msg) => {
            if (msg.toolCall) {
              for (const fc of msg.toolCall.functionCalls) {
                if (fc.name === 'switchToSam') {
                  // Trigger Auditory and Visual handoff
                  if (audioContexts.current.output) {
                    playHandoffChime(audioContexts.current.output);
                  }
                  
                  setIsSwitching(true);
                  setTranscripts(prev => [...prev, { role: 'system', text: 'URGENT: ELEVATING TO LEAD DISPATCH...', timestamp: new Date() }]);
                  
                  setTimeout(() => {
                    setCurrentPersona(AgentPersona.SAM);
                    setIsSwitching(false);
                  }, 2200);
                }
                session.sendToolResponse({ functionResponses: { id: fc.id, name: fc.name, response: { result: "ok" } } });
              }
            }
            if (msg.serverContent?.inputTranscription?.text) {
              const text = msg.serverContent.inputTranscription.text;
              setTranscripts(prev => [...prev, { role: 'user', text, timestamp: new Date() }]);
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
      sessionRef.current = session;
    } catch (e) { console.error(e); }
  };

  const isChloe = currentPersona === AgentPersona.CHLOE;

  return (
    <div className={`min-h-screen relative flex flex-col items-center p-6 sm:p-12 transition-all duration-1000 ${isChloe ? 'bg-emerald-50' : 'bg-rose-50'}`}>
      
      {/* Dynamic Background Blobs */}
      <div className="mesh-gradient">
        <div className={`mesh-blob bg-${isChloe ? 'emerald-300' : 'rose-300'} top-[-10%] left-[-10%] transition-colors duration-1000`} />
        <div className={`mesh-blob bg-${isChloe ? 'cyan-300' : 'orange-300'} bottom-[-10%] right-[-10%] animation-delay-2000 transition-colors duration-1000`} />
      </div>

      <header className="w-full max-w-7xl flex items-center justify-between mb-12 relative z-10">
        <div className="flex items-center gap-6">
          <div className={`w-16 h-16 ${isChloe ? 'bg-emerald-600 shadow-emerald-200' : 'bg-rose-600 shadow-rose-200'} rounded-[2rem] flex items-center justify-center shadow-2xl transition-all duration-700`}>
            <Leaf className="w-9 h-9 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-[900] tracking-tighter text-slate-900 uppercase italic">Green Choice</h1>
            <div className="flex items-center gap-3">
              <span className={`text-[11px] font-black ${isChloe ? 'text-emerald-600' : 'text-rose-600'} tracking-[0.2em] uppercase transition-colors duration-700`}>GTA Dispatch Hub</span>
              <div className="w-1.5 h-1.5 bg-slate-300 rounded-full" />
              <div className="flex items-center gap-1.5">
                 <Signal className={`w-3.5 h-3.5 ${isConnected ? 'text-blue-500 animate-pulse' : 'text-slate-300'}`} />
                 <span className="text-[10px] font-black text-slate-400 uppercase">Secure Link v2.6</span>
              </div>
            </div>
          </div>
        </div>

        <div className="hidden lg:flex items-center gap-6">
          <div className="glass-container px-6 py-3 rounded-2xl flex items-center gap-3">
             <div className={`w-2 h-2 rounded-full ${isConnected ? (isChloe ? 'bg-emerald-500' : 'bg-rose-500') + ' animate-ping' : 'bg-slate-300'} transition-colors duration-700`} />
             <span className="text-xs font-black text-slate-700 uppercase tracking-widest">{isConnected ? 'Link Established' : 'System Standby'}</span>
          </div>
          <div className="bg-white p-2 rounded-2xl shadow-xl">
             <div className="flex items-center gap-2 px-3 py-1 bg-slate-50 rounded-xl border border-slate-100">
               <MapPin className={`w-4 h-4 ${isChloe ? 'text-emerald-500' : 'text-rose-500'} transition-colors duration-700`} />
               <span className="text-[10px] font-black text-slate-600 uppercase">Toronto Regional Hub</span>
             </div>
          </div>
        </div>
      </header>

      <main className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-12 gap-10 items-start relative z-10">
        
        {/* Left Stats Section */}
        <div className="lg:col-span-3 space-y-6">
          <div className="flex flex-col gap-4">
             <HighVisibilityMetric 
               icon={Radio} 
               label="Signal Mode" 
               value="Native Audio" 
               colorClass={isChloe ? 'bg-emerald-500' : 'bg-rose-500'} 
             />
             <HighVisibilityMetric 
               icon={Volume2} 
               label="Audio Transmit" 
               value="Real-time PCM" 
               colorClass={isChloe ? 'bg-cyan-500' : 'bg-orange-500'} 
             />
             <HighVisibilityMetric 
               icon={ShieldCheck} 
               label="Privacy" 
               value="E2E Encrypted" 
               colorClass={isChloe ? 'bg-indigo-500' : 'bg-purple-500'} 
             />
          </div>

          <div className={`p-8 rounded-[2.5rem] transition-all duration-1000 ${isChloe ? 'bg-gradient-to-br from-emerald-600 to-teal-700' : 'bg-gradient-to-br from-rose-600 to-orange-700'} text-white shadow-2xl relative overflow-hidden group`}>
             <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:scale-125 transition-transform">
               <Sparkles className="w-24 h-24" />
             </div>
             <div className="relative z-10">
               <div className="flex items-center gap-2 mb-4">
                 <Zap className="w-5 h-5 text-yellow-300 fill-yellow-300" />
                 <span className="text-xs font-black uppercase tracking-widest">Active Intelligence</span>
               </div>
               <p className="text-sm font-bold leading-relaxed opacity-90 italic">
                 {isChloe 
                  ? "Chloe is prioritizing residential leads. Detecting distress signals will trigger immediate Sam escalation." 
                  : "Sam is now leading the priority emergency dispatch. Emergency services are on standby."}
               </p>
             </div>
          </div>
        </div>

        {/* Center Portal Section */}
        <div className={`lg:col-span-6 glass-container rounded-[4rem] p-12 flex flex-col items-center justify-center relative overflow-hidden transition-all duration-1000 min-h-[640px] shadow-[0_50px_100px_-20px_rgba(0,0,0,0.1)] ${isChloe ? 'agent-glow-chloe' : 'agent-glow-sam'} ${isSwitching ? 'scale-[0.98] ring-4 ring-white' : ''}`}>
          
          <div className="absolute top-12 flex items-center gap-3 px-6 py-2 bg-white rounded-full shadow-lg border border-slate-100">
            <div className={`w-2.5 h-2.5 rounded-full ${isConnected ? (isChloe ? 'bg-emerald-500' : 'bg-rose-500') : 'bg-slate-300'} ${isConnected ? 'animate-pulse' : ''} transition-colors duration-700`} />
            <span className="text-[11px] font-black uppercase text-slate-800 tracking-[0.2em]">Voice Gateway Active</span>
          </div>

          <AudioVisualizerPortal volume={volumeLevel} persona={currentPersona} isActive={isConnected} isSwitching={isSwitching} />

          <div className="mt-12 text-center">
            <h2 className={`text-6xl font-[1000] tracking-tighter text-slate-900 italic drop-shadow-sm transition-all duration-500 ${isSwitching ? 'blur-sm opacity-50' : ''}`}>
              {isChloe ? 'Chloe' : 'Sam'}
            </h2>
            <div className={`inline-flex items-center gap-2 mt-4 px-4 py-1.5 rounded-xl text-[11px] font-black uppercase tracking-[0.2em] transition-all duration-1000 ${isChloe ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'} ${isSwitching ? 'animate-bounce' : ''}`}>
              {isChloe ? <ThermometerSun className="w-4 h-4" /> : <ShieldAlert className="w-4 h-4" />}
              {isChloe ? 'Efficiency Specialist' : 'Priority Dispatch Lead'}
            </div>
          </div>

          <div className="w-full max-w-sm mt-14">
             <button
               onClick={isConnected ? disconnect : connect}
               className={`w-full py-6 rounded-[2rem] font-black text-lg tracking-[0.1em] uppercase transition-all duration-500 flex items-center justify-center gap-4 shadow-[0_20px_40px_-10px_rgba(0,0,0,0.1)] hover:-translate-y-2 hover:shadow-[0_30px_60px_-10px_rgba(0,0,0,0.2)] active:translate-y-0 relative overflow-hidden ${
                 isConnected 
                   ? 'bg-slate-900 text-white' 
                   : isChloe 
                     ? 'bg-emerald-600 text-white shadow-emerald-200' 
                     : 'bg-rose-600 text-white shadow-rose-200'
               } ${isSwitching ? 'opacity-20 cursor-not-allowed' : ''}`}
               disabled={isSwitching}
             >
               {isConnected ? <PhoneOff className="w-6 h-6" /> : <Phone className="w-6 h-6 animate-pulse" />}
               <span>{isConnected ? 'Kill Link' : 'Secure Call'}</span>
               {!isConnected && <div className="absolute inset-0 shimmer opacity-30" />}
             </button>
          </div>
        </div>

        {/* Right Log Section */}
        <div className="lg:col-span-3">
          <div className="glass-container rounded-[3rem] h-[640px] flex flex-col overflow-hidden shadow-2xl">
            <div className="p-8 border-b border-slate-100/50 bg-white/40 flex items-center justify-between">
               <h3 className="text-xs font-black text-slate-500 uppercase tracking-widest">Live Metadata</h3>
               <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-50 text-blue-600 text-[10px] font-black">
                 <Signal className="w-3 h-3" />
                 ECHO SYNC
               </div>
            </div>
            
            <div className="flex-1 overflow-y-auto p-8 space-y-8 scrollbar-hide">
               {transcripts.length === 0 ? (
                 <div className="h-full flex flex-col items-center justify-center text-slate-300 text-center gap-6">
                    <Activity className="w-16 h-16 opacity-10 animate-pulse" />
                    <p className="text-[11px] font-black uppercase tracking-[0.15em] leading-relaxed max-w-[150px]">Waiting for voice authentication handshake...</p>
                 </div>
               ) : (
                 transcripts.map((t, i) => (
                   <div key={i} className={`animate-in slide-in-from-bottom-4 duration-700 ${t.role === 'system' ? 'opacity-100' : ''}`}>
                      <div className="flex items-center justify-between mb-3">
                        <span className={`text-[10px] font-black ${t.role === 'system' ? 'text-red-600' : (isChloe ? 'text-emerald-600' : 'text-rose-600')} uppercase tracking-widest transition-colors`}>
                          {t.role === 'system' ? 'SYSTEM OVERRIDE' : 'External Source'}
                        </span>
                        <span className="text-[9px] font-bold text-slate-400 bg-white px-2 py-0.5 rounded-md shadow-sm">{t.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                      </div>
                      <div className={`text-sm font-bold leading-relaxed p-5 rounded-[2rem] shadow-sm border relative transition-all duration-500 ${t.role === 'system' ? 'bg-red-50 text-red-700 border-red-200 animate-pulse' : 'bg-white/80 text-slate-700 border-slate-50'}`}>
                        {t.text}
                        <div className={`absolute top-4 -left-1 w-1.5 h-6 rounded-full ${t.role === 'system' ? 'bg-red-600' : (isChloe ? 'bg-emerald-400' : 'bg-rose-400')}`} />
                      </div>
                   </div>
                 ))
               )}
            </div>

            <div className="p-8 bg-white/60 border-t border-slate-100">
              <div className="flex items-center justify-between text-[11px] font-black text-slate-500 uppercase tracking-widest mb-3">
                <span>Buffer Integrity</span>
                <span className="text-emerald-600">99.8%</span>
              </div>
              <div className="w-full h-2.5 bg-slate-200/50 rounded-full overflow-hidden">
                <div className={`h-full ${isChloe ? 'bg-emerald-500' : 'bg-rose-500'} w-[90%] transition-all duration-1000`} />
              </div>
            </div>
          </div>
        </div>

      </main>

      <footer className="mt-16 w-full max-w-7xl flex flex-col sm:flex-row items-center justify-between gap-8 border-t border-slate-200/50 pt-10 relative z-10">
        <div className="flex items-center gap-12">
           <div className="flex items-center gap-3">
             <div className="w-3 h-3 bg-emerald-500 rounded-full shadow-lg shadow-emerald-500/50" />
             <span className="text-[11px] font-black text-slate-400 uppercase tracking-widest">GTA Core Security</span>
           </div>
           <div className="flex items-center gap-3">
             <div className="w-3 h-3 bg-blue-500 rounded-full shadow-lg shadow-blue-500/50" />
             <span className="text-[11px] font-black text-slate-400 uppercase tracking-widest">AI Engine Tier 1</span>
           </div>
        </div>
        <div className="text-[11px] font-black text-slate-500 uppercase tracking-[0.3em] text-center italic">
          © 2026 Green Choice Heating & Cooling | Enterprise Dispatch System
        </div>
      </footer>

      {/* Extreme Priority Switcher Overlay (Enhanced Transitions) */}
      {isSwitching && (
        <div className="fixed inset-0 z-50 bg-rose-600/95 backdrop-blur-[40px] flex flex-col items-center justify-center animate-in fade-in zoom-in duration-300 p-8">
           <div className="relative group max-w-lg w-full">
              {/* White Flash Effect */}
              <div className="absolute inset-0 bg-white blur-[150px] opacity-30 animate-pulse" />
              
              <div className="relative bg-white/10 p-16 rounded-[4rem] border-2 border-white/20 shadow-[0_0_80px_rgba(255,255,255,0.1)] flex flex-col items-center gap-10">
                 <div className="w-32 h-32 bg-white rounded-full flex items-center justify-center animate-bounce shadow-[0_0_50px_rgba(255,255,255,0.5)]">
                    <ShieldAlert className="w-16 h-16 text-rose-600" />
                 </div>
                 <div className="text-center space-y-4">
                   <h2 className="text-5xl font-[1000] text-white tracking-tighter italic animate-pulse">PRIORITY RED ALERT</h2>
                   <p className="text-rose-100 font-black uppercase tracking-[0.4em] text-xs">Elevating Link to Priority Sam</p>
                 </div>
                 
                 {/* Progress Bars */}
                 <div className="flex gap-3 w-full">
                   {[0,1,2,3].map(i => (
                     <div key={i} className="flex-1 h-3 bg-white/10 rounded-full overflow-hidden">
                       <div className="w-full h-full bg-white animate-[loading_2s_infinite] shadow-[0_0_20px_white]" style={{ animationDelay: `${i*0.25}s` }} />
                     </div>
                   ))}
                 </div>

                 <div className="flex items-center gap-3 text-white/60 text-[10px] font-black uppercase tracking-widest animate-pulse">
                    <Radio className="w-3 h-3" />
                    Bypassing Routine Latency...
                 </div>
              </div>
           </div>
        </div>
      )}
    </div>
  );
}
