import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type, FunctionDeclaration } from "@google/genai";
import { AgentPersona, MessageLog, BlobData } from './types';
import { 
  Phone, PhoneOff, Leaf, Activity, 
  ArrowRightLeft, ThermometerSun, ShieldAlert, 
  CheckCircle2, MapPin, Signal, Radio, 
  ChevronRight, AlertCircle, Zap
} from 'lucide-react';

// --- Constants & Config ---
const getApiKey = (): string => {
  try {
    // @ts-ignore
    if (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_KEY) {
      // @ts-ignore
      return import.meta.env.VITE_API_KEY;
    }
    if (typeof process !== 'undefined' && process.env && process.env.API_KEY) {
      return process.env.API_KEY;
    }
  } catch (e) {
    console.warn("Error reading environment variables", e);
  }
  return '';
};

const API_KEY = getApiKey();
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

// --- Helpers ---
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

// --- Sub-Components ---

const AudioAura: React.FC<{ volume: number; persona: AgentPersona; isActive: boolean }> = ({ volume, persona, isActive }) => {
  const isChloe = persona === AgentPersona.CHLOE;
  const color = isChloe ? 'from-emerald-400 to-cyan-500' : 'from-red-500 to-orange-600';
  const scale = 1 + (volume * 1.5);

  return (
    <div className="relative flex items-center justify-center w-64 h-64">
      {/* Outer Glow Rings */}
      <div 
        className={`absolute inset-0 rounded-full bg-gradient-to-tr ${color} opacity-10 blur-2xl transition-all duration-300`}
        style={{ transform: `scale(${scale * 1.2})` }}
      />
      <div 
        className={`absolute inset-4 rounded-full border-2 border-white/10 audio-ring transition-all duration-300`}
        style={{ transform: `scale(${scale * 1.1})` }}
      />
      
      {/* Active Frequency Bars (Circular) */}
      <div className="absolute inset-0 flex items-center justify-center">
        {Array.from({ length: 24 }).map((_, i) => (
          <div
            key={i}
            className={`absolute w-1 rounded-full bg-gradient-to-t ${color} transition-all duration-75`}
            style={{
              height: isActive ? `${Math.max(10, volume * 100 * (0.5 + Math.random()))}%` : '4px',
              transform: `rotate(${i * 15}deg) translateY(-85px)`,
              opacity: isActive ? 0.8 : 0.2
            }}
          />
        ))}
      </div>

      {/* Center Image Container */}
      <div className={`relative w-40 h-40 rounded-full border-4 ${isChloe ? 'border-emerald-500/50' : 'border-red-500/50'} overflow-hidden shadow-2xl bg-gray-900`}>
        <img 
          src={isChloe 
            ? "https://images.unsplash.com/photo-1580489944761-15a19d654956?auto=format&fit=crop&q=80&w=300&h=300" 
            : "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&q=80&w=300&h=300"
          } 
          className={`w-full h-full object-cover transition-all duration-700 ${isActive ? 'scale-105' : 'scale-100 grayscale'}`}
          alt="Agent"
        />
        {!isActive && <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" />}
      </div>
    </div>
  );
};

const StatusMetric: React.FC<{ icon: any; label: string; value: string; color: string }> = ({ icon: Icon, label, value, color }) => (
  <div className="flex flex-col gap-1 p-3 rounded-xl bg-white/5 border border-white/10">
    <div className="flex items-center gap-2 text-[10px] font-bold text-gray-400 uppercase tracking-tighter">
      <Icon className={`w-3 h-3 ${color}`} />
      {label}
    </div>
    <div className="text-sm font-bold text-gray-200">{value}</div>
  </div>
);

// --- Main Application ---

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [currentPersona, setCurrentPersona] = useState<AgentPersona>(AgentPersona.CHLOE);
  const [isSwitching, setIsSwitching] = useState(false);
  const [volumeLevel, setVolumeLevel] = useState(0);
  const [transcripts, setTranscripts] = useState<MessageLog[]>([]);
  const [isSurveyMode, setIsSurveyMode] = useState(false);

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
                  setIsSwitching(true);
                  setTimeout(() => {
                    setCurrentPersona(AgentPersona.SAM);
                    setIsSwitching(false);
                  }, 1500);
                } else if (fc.name === 'startSurvey') {
                  setIsSurveyMode(true);
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

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col items-center p-6 selection:bg-emerald-500/30">
      
      {/* HUD Header */}
      <header className="w-full max-w-5xl flex items-center justify-between mb-8 px-4">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-emerald-600 rounded-xl flex items-center justify-center shadow-lg shadow-emerald-900/20">
            <Leaf className="w-7 h-7 text-white" />
          </div>
          <div className="hidden sm:block">
            <h1 className="text-xl font-extrabold tracking-tight text-white uppercase">Green Choice</h1>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-bold text-emerald-500 tracking-widest uppercase">HVAC Dispatch GTA</span>
              <div className="h-1 w-1 bg-gray-600 rounded-full" />
              <span className="text-[10px] font-bold text-gray-500">v2.5 Live</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex flex-col items-end">
             <div className="flex items-center gap-2 text-xs font-bold text-gray-400">
               <MapPin className="w-3 h-3 text-emerald-500" />
               East York HQ
             </div>
             <div className="text-[10px] text-gray-600 font-bold uppercase tracking-tighter">Toronto, ON</div>
          </div>
          <div className={`flex items-center gap-3 px-4 py-2 rounded-xl glass-card border ${isConnected ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-red-500/30 bg-red-500/5'}`}>
            <Signal className={`w-4 h-4 ${isConnected ? 'text-emerald-400' : 'text-red-400'} animate-pulse`} />
            <span className="text-xs font-black tracking-widest uppercase">{isConnected ? 'Link Active' : 'Disconnected'}</span>
          </div>
        </div>
      </header>

      {/* Main Control Center */}
      <main className="w-full max-w-5xl grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Side: Diagnostics & Status */}
        <div className="lg:col-span-3 space-y-4">
          <div className="glass-card p-5 rounded-3xl space-y-6">
            <h3 className="text-xs font-black text-gray-500 uppercase tracking-widest flex items-center gap-2">
              <Zap className="w-3 h-3 text-emerald-500" />
              System Metrics
            </h3>
            <div className="grid grid-cols-1 gap-3">
              <StatusMetric icon={Radio} label="Carrier" value="Native Audio" color="text-cyan-400" />
              <StatusMetric icon={Activity} label="Latency" value="142ms" color="text-emerald-400" />
              <StatusMetric icon={ThermometerSun} label="Heat Index" value="High Load" color="text-orange-400" />
            </div>
          </div>

          <div className="glass-card p-5 rounded-3xl border-l-4 border-l-emerald-500">
            <h4 className="text-xs font-bold text-emerald-500 mb-2">Lead Insights</h4>
            <p className="text-[11px] text-gray-400 leading-relaxed font-medium">
              "Chloe is currently qualifying residential rebate leads in East York tier-1 zones."
            </p>
          </div>
        </div>

        {/* Center Side: Agent Visualizer */}
        <div className={`lg:col-span-6 glass-card rounded-[3rem] p-8 flex flex-col items-center justify-center relative overflow-hidden transition-all duration-700 ${currentPersona === AgentPersona.SAM ? 'sam-active bg-red-950/10' : 'chloe-active bg-emerald-950/10'}`}>
          {currentPersona === AgentPersona.SAM && <div className="scanline-effect" />}
          
          <div className="absolute top-8 left-8 flex items-center gap-2 px-3 py-1 bg-white/5 rounded-full border border-white/10 backdrop-blur-md">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-400 animate-pulse' : 'bg-gray-600'}`} />
            <span className="text-[10px] font-black uppercase text-gray-300 tracking-tighter">Session Encrypted</span>
          </div>

          <AudioAura volume={volumeLevel} persona={currentPersona} isActive={isConnected} />

          <div className="mt-8 text-center">
            <div className="flex items-center justify-center gap-3">
               {currentPersona === AgentPersona.SAM ? <ShieldAlert className="w-6 h-6 text-red-500" /> : <ThermometerSun className="w-6 h-6 text-emerald-500" />}
               <h2 className="text-4xl font-black tracking-tight text-white">
                {currentPersona === AgentPersona.CHLOE ? 'CHLOE' : 'SAM'}
               </h2>
            </div>
            <p className={`text-sm font-bold uppercase tracking-[0.2em] mt-2 transition-colors duration-500 ${currentPersona === AgentPersona.CHLOE ? 'text-emerald-400' : 'text-red-500'}`}>
              {currentPersona === AgentPersona.CHLOE ? 'Service Concierge' : 'Emergency Lead'}
            </p>
          </div>

          <div className="w-full mt-10 space-y-4">
             <button
               onClick={isConnected ? disconnect : connect}
               className={`w-full py-5 rounded-2xl font-black text-lg transition-all duration-500 flex items-center justify-center gap-3 group relative overflow-hidden ${
                 isConnected 
                   ? 'bg-white text-slate-950 hover:bg-gray-200' 
                   : currentPersona === AgentPersona.CHLOE 
                     ? 'bg-emerald-600 text-white hover:bg-emerald-500 hover:scale-[1.02]' 
                     : 'bg-red-600 text-white hover:bg-red-500'
               }`}
             >
               {isConnected ? <PhoneOff className="w-6 h-6" /> : <Phone className="w-6 h-6" />}
               {isConnected ? 'TERMINATE SESSION' : 'ESTABLISH LINK'}
               {isConnected && <div className="absolute inset-0 bg-white/10 group-active:bg-black/10" />}
             </button>
          </div>
        </div>

        {/* Right Side: Log & Comms */}
        <div className="lg:col-span-3 space-y-4">
          <div className="glass-card rounded-3xl h-[480px] flex flex-col">
            <div className="p-4 border-b border-white/10 flex items-center justify-between">
               <h3 className="text-xs font-black text-gray-500 uppercase tracking-widest">Live Comm Log</h3>
               <div className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 text-[10px] font-bold">Auto-Sync</div>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-4 font-medium">
               {transcripts.length === 0 ? (
                 <div className="h-full flex flex-col items-center justify-center text-gray-600 text-center px-4">
                    <Activity className="w-10 h-10 mb-2 opacity-20" />
                    <p className="text-xs italic uppercase tracking-tighter">Waiting for voice input handshake...</p>
                 </div>
               ) : (
                 transcripts.map((t, i) => (
                   <div key={i} className="flex flex-col gap-1 group">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-black text-emerald-500 uppercase tracking-tighter">Field Tech</span>
                        <span className="text-[9px] text-gray-600">{t.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                      </div>
                      <p className="text-xs text-gray-300 leading-relaxed bg-white/5 p-3 rounded-xl border border-white/5 group-hover:border-emerald-500/30 transition-colors">
                        {t.text}
                      </p>
                   </div>
                 ))
               )}
            </div>
            <div className="p-4 border-t border-white/10">
              <div className="flex items-center justify-between text-[10px] font-bold text-gray-600 uppercase">
                <span>Buffer Status</span>
                <span className="text-emerald-500">Optimum</span>
              </div>
            </div>
          </div>
        </div>

      </main>

      <footer className="mt-12 w-full max-w-5xl border-t border-white/5 pt-6 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-6">
           <div className="flex items-center gap-2">
             <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full" />
             <span className="text-[10px] font-bold text-gray-500 uppercase">All Systems Nominal</span>
           </div>
           <div className="flex items-center gap-2">
             <div className="w-1.5 h-1.5 bg-cyan-500 rounded-full" />
             <span className="text-[10px] font-bold text-gray-500 uppercase">Encrypted Dispatch</span>
           </div>
        </div>
        <p className="text-[10px] font-black text-gray-600 uppercase tracking-widest">
          © 2026 Green Choice Heating & Cooling | Greater Toronto Area
        </p>
      </footer>

      {isSwitching && (
        <div className="fixed inset-0 z-50 bg-slate-950/90 backdrop-blur-xl flex flex-col items-center justify-center animate-in fade-in duration-300">
           <div className="relative">
              <div className="absolute inset-0 bg-red-600 blur-3xl opacity-20 animate-pulse" />
              <div className="relative bg-red-600/10 p-10 rounded-[2.5rem] border border-red-500/50 flex flex-col items-center gap-6">
                 <ShieldAlert className="w-20 h-20 text-red-500 animate-bounce" />
                 <div className="text-center">
                   <h2 className="text-3xl font-black text-white italic tracking-tight">ELEVATING PRIORITY</h2>
                   <p className="text-red-400 font-bold uppercase tracking-[0.3em] mt-2 text-sm">Transferring to Lead Dispatch Sam</p>
                 </div>
                 <div className="flex gap-1">
                   {[0,1,2].map(i => <div key={i} className="w-12 h-1 bg-red-600/30 overflow-hidden"><div className="w-full h-full bg-red-500 animate-[loading_1.5s_infinite]" style={{ animationDelay: `${i*0.2}s` }} /></div>)}
                 </div>
              </div>
           </div>
        </div>
      )}
    </div>
  );
}