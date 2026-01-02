import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type, FunctionDeclaration } from "@google/genai";
import { AgentPersona, MessageLog, BlobData } from './types';
import { Phone, PhoneOff, AlertTriangle, Leaf, History, Activity, ArrowRightLeft, ThermometerSun, ShieldAlert, CheckCircle2 } from 'lucide-react';

// --- Constants & Config ---

// Safe API Key retrieval
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

// --- System Instructions ---
const SYSTEM_INSTRUCTION = `
You are the Voice AI system for 'Green Choice Heating & Cooling' in East York.
You have two distinct personas. You start as CHLOE.

PERSONA 1: CHLOE (Front-Desk & Rebate Specialist)
- Tone: Warm, energetic, knowledgeable, and patient.
- Role: Explain the "Home Renovation Savings Program" (HRS) and book routine assessments.
- KNOWLEDGE BASE (Strict Adherence Required):
  1. **NON-GAS HEATING (Current system is Electric, Oil, Propane, or Wood):**
     - **Air Source Heat Pump:** Rebate is **$1,250 per ton**, up to a maximum of **$7,500**.
     - **Ground Source Heat Pump:** Rebate is **$2,000 per ton**, up to a maximum of **$12,000**.
  2. **ENBRIDGE GAS CUSTOMERS (Current system is Natural Gas):**
     - **Air Source Heat Pump:** Rebate is **$500 per ton**, up to a maximum of **$2,000**.
     - **Ground Source Heat Pump:** Flat rebate of **$3,000**.
  3. **RENTALS:**
     - Rentals follow the Enbridge Gas tier ($500/ton up to $2,000) regardless of fuel type.

- PROTOCOL:
  1. Greet warmly: "Thanks for calling Green Choice! Are you looking for a repair or information on the 2026 heat pump rebates?"
  2. If Rebates: You MUST ask: "What fuel specifically heats your home right now? Natural gas, electricity, or oil?"
  3. Calculate and confirm their potential savings based on the logic above.
  4. Collect Name/Phone to book an assessment.

PERSONA 2: SAM (Emergency Dispatcher)
- Tone: Serious, rapid, authoritative, reassuring.
- Role: Immediate safety and dispatch.
- Triggers: "gas smell", "no heat", "leaking water", "loud banging", "smoke", "carbon monoxide".
- PROTOCOL:
  1. IF "Gas Smell" is mentioned: "For your safety, hang up immediately, leave the house, and call 911."
  2. FOR OTHER EMERGENCIES: "I'm dispatching a tech. I need your address now. We have a 4-hour response guarantee."

LOGIC FLOW:
1. Listen for emergency keywords. 
2. IF keywords detected -> Say: "This sounds urgent. Connecting you to Sam, our emergency dispatcher." -> Call tool \`switchToSam\`.
3. IF routine -> Continue as Chloe.
4. IF booking complete -> Call tool \`startSurvey\`.
`;

// --- Tool Definitions ---
const switchToSamTool: FunctionDeclaration = {
  name: 'switchToSam',
  description: 'Triggers the persona switch to Sam when an emergency keyword is detected.',
  parameters: { type: Type.OBJECT, properties: {} },
};

const startSurveyTool: FunctionDeclaration = {
  name: 'startSurvey',
  description: 'Triggers the satisfaction survey mode after a call is wrapped up.',
  parameters: { type: Type.OBJECT, properties: {} },
};

const tools = [{ functionDeclarations: [switchToSamTool, startSurveyTool] }];

// --- Audio Helpers ---
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
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
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

function playHandoffSound(ctx: AudioContext) {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.connect(gain);
  gain.connect(ctx.destination);
  const now = ctx.currentTime;
  osc.type = 'sawtooth';
  osc.frequency.setValueAtTime(440, now);
  osc.frequency.linearRampToValueAtTime(880, now + 0.1);
  osc.frequency.setValueAtTime(880, now + 0.2);
  osc.frequency.exponentialRampToValueAtTime(1760, now + 0.4);
  gain.gain.setValueAtTime(0.05, now);
  gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
  osc.start(now);
  osc.stop(now + 0.5);
}

// --- Components ---

const Visualizer: React.FC<{ volume: number; persona: AgentPersona; isActive: boolean }> = ({ volume, persona, isActive }) => {
  const bars = 12;
  const isChloe = persona === AgentPersona.CHLOE;
  
  return (
    <div className="flex items-center justify-center space-x-1.5 h-16 w-full px-8">
      {isActive ? Array.from({ length: bars }).map((_, i) => {
        // Create a wave effect
        const wave = Math.sin(i * 0.5 + Date.now() / 100) * 0.5 + 0.5;
        const height = Math.max(15, Math.min(100, volume * (150 + i * 10) * wave + 20)); 
        const colorClass = isChloe ? 'bg-emerald-400' : 'bg-red-500';
        
        return (
          <div
            key={i}
            className={`w-2 rounded-full transition-all duration-75 ${colorClass} shadow-[0_0_10px_rgba(0,0,0,0.1)]`}
            style={{ height: `${height}%` }}
          />
        );
      }) : (
        <div className="flex items-center space-x-2 text-gray-400 animate-pulse">
           <div className="w-2 h-2 bg-gray-300 rounded-full"></div>
           <div className="w-2 h-2 bg-gray-300 rounded-full animation-delay-200"></div>
           <div className="w-2 h-2 bg-gray-300 rounded-full animation-delay-400"></div>
        </div>
      )}
    </div>
  );
};

const ChatBubble: React.FC<{ msg: MessageLog; persona: AgentPersona }> = ({ msg, persona }) => {
  const isUser = msg.role === 'user';
  const isChloe = persona === AgentPersona.CHLOE;
  const agentColor = isChloe ? 'bg-emerald-50 text-emerald-900 border-emerald-100' : 'bg-red-50 text-red-900 border-red-100';

  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-4 animate-in fade-in slide-in-from-bottom-2 duration-300`}>
      <div className={`max-w-[85%] rounded-2xl px-5 py-3 text-sm font-medium shadow-sm ${
        isUser 
          ? 'bg-gray-800 text-white rounded-br-none' 
          : `${agentColor} border rounded-bl-none`
      }`}>
        {msg.text}
      </div>
    </div>
  );
};

// --- Main App ---

export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [currentPersona, setCurrentPersona] = useState<AgentPersona>(AgentPersona.CHLOE);
  const [isSwitching, setIsSwitching] = useState(false);
  const [volumeLevel, setVolumeLevel] = useState(0);
  const [transcripts, setTranscripts] = useState<MessageLog[]>([]);
  const [isSurveyMode, setIsSurveyMode] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const nextStartTime = useRef<number>(0);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const inputSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sessionRef = useRef<any>(null);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  useEffect(() => {
    return () => disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const disconnect = useCallback(() => {
    sessionRef.current = null;
    if (inputAudioContextRef.current) inputAudioContextRef.current.close();
    if (outputAudioContextRef.current) outputAudioContextRef.current.close();
    if (processorRef.current) processorRef.current.disconnect();
    if (inputSourceRef.current) inputSourceRef.current.disconnect();
    sourcesRef.current.forEach(source => { try { source.stop(); } catch(e) {} });
    sourcesRef.current.clear();
    setIsConnected(false);
    setVolumeLevel(0);
    setIsSwitching(false);
  }, []);

  const connect = async () => {
    setErrorMsg(null);
    try {
      if (!API_KEY) throw new Error("API Key not found");

      const InputContextClass = (window.AudioContext || (window as any).webkitAudioContext);
      const inputCtx = new InputContextClass({ sampleRate: 16000 });
      const outputCtx = new InputContextClass({ sampleRate: 24000 });
      inputAudioContextRef.current = inputCtx;
      outputAudioContextRef.current = outputCtx;
      const outputNode = outputCtx.createGain();
      outputNode.connect(outputCtx.destination);

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ai = new GoogleGenAI({ apiKey: API_KEY });

      let resolveSession: (s: any) => void;
      const sessionPromise = new Promise<any>(resolve => { resolveSession = resolve; });

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
            resolveSession(session);
            setIsConnected(true);
            setCurrentPersona(AgentPersona.CHLOE);
            setIsSurveyMode(false);
            setTranscripts([]);
            setIsSwitching(false);

            const source = inputCtx.createMediaStreamSource(stream);
            inputSourceRef.current = source;
            const processor = inputCtx.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;

            processor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              let sum = 0;
              for(let i=0; i<inputData.length; i++) sum += inputData[i] * inputData[i];
              const rms = Math.sqrt(sum / inputData.length);
              setVolumeLevel(prev => Math.max(rms * 5, prev * 0.9)); 

              sessionPromise.then(sess => sess.sendRealtimeInput({ media: createBlob(inputData) }));
            };
            source.connect(processor);
            processor.connect(inputCtx.destination);
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.toolCall) {
              for (const fc of message.toolCall.functionCalls) {
                if (fc.name === 'switchToSam') {
                  if (outputAudioContextRef.current) playHandoffSound(outputAudioContextRef.current);
                  setIsSwitching(true);
                  setTranscripts(prev => [...prev, { role: 'system', text: 'Transferring to Priority Dispatch...', timestamp: new Date() }]);
                  setTimeout(() => {
                    setCurrentPersona(AgentPersona.SAM);
                    setIsSwitching(false);
                  }, 2000);
                } else if (fc.name === 'startSurvey') {
                  setIsSurveyMode(true);
                  setTranscripts(prev => [...prev, { role: 'system', text: 'Call Complete. Entering Survey Mode.', timestamp: new Date() }]);
                }
                sessionPromise.then(sess => sess.sendToolResponse({ functionResponses: { id: fc.id, name: fc.name, response: { result: "ok" } } }));
              }
            }

            if (message.serverContent?.inputTranscription?.text) {
               const text = message.serverContent.inputTranscription.text;
               setTranscripts(prev => {
                 const last = prev[prev.length - 1];
                 if (last && last.role === 'user') return [...prev.slice(0, -1), { ...last, text: last.text + text }];
                 return [...prev, { role: 'user', text, timestamp: new Date() }];
               });
            }

            const base64Audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (base64Audio && outputAudioContextRef.current) {
              const ctx = outputAudioContextRef.current;
              nextStartTime.current = Math.max(nextStartTime.current, ctx.currentTime);
              const audioBuffer = await decodeAudioData(decodeAudio(base64Audio), ctx, 24000, 1);
              const source = ctx.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(outputNode);
              source.addEventListener('ended', () => sourcesRef.current.delete(source));
              source.start(nextStartTime.current);
              nextStartTime.current += audioBuffer.duration;
              sourcesRef.current.add(source);
            }
            
            if (message.serverContent?.interrupted) {
              sourcesRef.current.forEach(s => s.stop());
              sourcesRef.current.clear();
              nextStartTime.current = 0;
            }
          },
          onclose: () => setIsConnected(false),
          onerror: (err) => {
            console.error(err);
            setErrorMsg("Connection lost. Please reconnect.");
            disconnect();
          }
        }
      });
      sessionRef.current = session;
    } catch (e) {
      console.error(e);
      setErrorMsg("Microphone access required.");
      disconnect();
    }
  };

  const handleToggleCall = () => (isConnected ? disconnect() : connect());
  
  const isChloe = currentPersona === AgentPersona.CHLOE;
  
  // Dynamic Backgrounds
  const getBackground = () => {
    if (isSwitching) return 'bg-gradient-to-br from-gray-100 to-gray-300';
    if (!isConnected) return 'bg-gradient-to-br from-emerald-50 to-teal-100';
    if (isChloe) return 'bg-gradient-to-br from-emerald-50 to-green-100';
    return 'bg-gradient-to-br from-red-50 to-orange-100';
  };

  if (!API_KEY) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 p-6 font-sans">
        <div className="max-w-md w-full bg-white p-8 rounded-2xl shadow-xl border border-gray-200 text-center">
          <AlertTriangle className="w-12 h-12 text-amber-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-gray-900">Setup Required</h2>
          <p className="text-gray-500 mt-2 mb-6">Please add <code className="bg-gray-100 px-2 py-1 rounded text-sm font-mono text-blue-600">VITE_API_KEY</code> to your environment variables.</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen ${getBackground()} transition-all duration-1000 flex flex-col items-center p-4 sm:p-8 font-sans`}>
      
      {/* Navbar */}
      <nav className="w-full max-w-2xl flex items-center justify-between mb-8 p-4 bg-white/60 backdrop-blur-md rounded-2xl border border-white/50 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="bg-emerald-600 p-2 rounded-lg text-white">
            <Leaf className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-800 leading-tight">Green Choice</h1>
            <p className="text-[10px] font-bold text-gray-400 tracking-widest uppercase">Heating & Cooling</p>
          </div>
        </div>
        <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold border ${isConnected ? 'bg-green-100 text-green-700 border-green-200' : 'bg-gray-100 text-gray-500 border-gray-200'}`}>
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
          {isConnected ? 'LIVE' : 'OFFLINE'}
        </div>
      </nav>

      {/* Main Agent Interface */}
      <div className="w-full max-w-2xl bg-white/80 backdrop-blur-xl rounded-[2.5rem] shadow-2xl overflow-hidden border border-white/50 relative">
        
        {/* Switching Overlay */}
        {isSwitching && (
          <div className="absolute inset-0 z-50 bg-white/95 flex flex-col items-center justify-center animate-in fade-in duration-300">
             <div className="bg-red-50 p-6 rounded-full mb-4 animate-bounce border-4 border-red-100">
                <ArrowRightLeft className="w-12 h-12 text-red-600" />
             </div>
             <h3 className="text-2xl font-bold text-gray-900 tracking-tight">Rerouting Call...</h3>
             <p className="text-sm text-gray-500 font-medium mt-2">Connecting to Priority Dispatch</p>
          </div>
        )}

        {/* Header / Avatar Section */}
        <div className={`relative w-full h-80 flex flex-col items-center justify-center transition-colors duration-700 ${isChloe ? 'bg-gradient-to-b from-emerald-50/50' : 'bg-gradient-to-b from-red-50/50'}`}>
          
          <div className="absolute top-6 right-6">
             {isSurveyMode ? (
               <span className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-100 text-blue-700 rounded-full text-xs font-bold border border-blue-200 shadow-sm">
                 <CheckCircle2 className="w-3 h-3" /> Survey Mode
               </span>
             ) : (
                <span className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold border shadow-sm transition-colors duration-500 ${isChloe ? 'bg-emerald-100 text-emerald-700 border-emerald-200' : 'bg-red-100 text-red-700 border-red-200'}`}>
                  {isChloe ? <ThermometerSun className="w-3 h-3" /> : <ShieldAlert className="w-3 h-3" />}
                  {isChloe ? 'Rebate Specialist' : 'Emergency Dispatch'}
                </span>
             )}
          </div>

          {/* Avatar */}
          <div className="relative group">
            <div className={`absolute -inset-1 rounded-full blur opacity-40 transition-colors duration-700 ${isChloe ? 'bg-emerald-400' : 'bg-red-600'}`}></div>
            <div className={`relative w-40 h-40 rounded-full border-[6px] shadow-2xl overflow-hidden bg-gray-100 transition-colors duration-700 ${isChloe ? 'border-white' : 'border-red-50'}`}>
              <img 
                src={isChloe 
                  ? "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?auto=format&fit=crop&q=80&w=200&h=200" 
                  : "https://images.unsplash.com/photo-1542909168-82c3e7fdca5c?auto=format&fit=crop&q=80&w=200&h=200"
                } 
                alt="Agent" 
                className={`w-full h-full object-cover transition-transform duration-700 ${isConnected ? 'scale-110' : 'scale-100 grayscale'}`}
              />
              {/* Overlay Flash on talk */}
              {isConnected && (
                <div className={`absolute inset-0 opacity-20 transition-colors duration-300 mix-blend-overlay ${isChloe ? 'bg-emerald-500' : 'bg-red-600'}`} 
                     style={{ opacity: Math.min(0.5, volumeLevel * 2) }} />
              )}
            </div>
          </div>

          <div className="mt-6 text-center">
            <h2 className="text-3xl font-bold text-gray-800 tracking-tight transition-all duration-300">
              {isChloe ? 'Chloe' : 'Sam'}
            </h2>
            <p className={`text-sm font-medium mt-1 transition-colors duration-300 ${isChloe ? 'text-emerald-600' : 'text-red-600'}`}>
              {isChloe ? 'Front Desk Agent' : 'Emergency Coordinator'}
            </p>
          </div>
        </div>

        {/* Visualizer & Status */}
        <div className="h-20 bg-white border-t border-b border-gray-100 flex flex-col items-center justify-center relative overflow-hidden">
           <Visualizer volume={volumeLevel} persona={currentPersona} isActive={isConnected} />
           {errorMsg && <div className="absolute bottom-1 text-[10px] font-bold text-red-500 bg-red-50 px-2 py-0.5 rounded-full">{errorMsg}</div>}
        </div>

        {/* Transcript Area */}
        <div className="h-72 bg-gray-50/50 p-4 overflow-y-auto scroll-smooth">
          {transcripts.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-300 space-y-3">
              <Activity className="w-12 h-12 opacity-20" />
              <p className="text-sm font-medium">Ready to start conversation</p>
            </div>
          ) : (
            <>
              {transcripts.map((msg, idx) => (
                msg.role === 'system' ? (
                  <div key={idx} className="flex justify-center my-4 animate-in fade-in zoom-in duration-300">
                    <span className="text-[10px] font-bold text-gray-400 bg-gray-100/80 backdrop-blur-sm px-3 py-1 rounded-full border border-gray-200 shadow-sm uppercase tracking-wide">
                      {msg.text}
                    </span>
                  </div>
                ) : (
                  <ChatBubble key={idx} msg={msg} persona={currentPersona} />
                )
              ))}
              <div className="h-4" /> {/* Spacer */}
            </>
          )}
        </div>

        {/* Footer Actions */}
        <div className="p-6 bg-white border-t border-gray-100 flex items-center justify-center">
          <button
            onClick={handleToggleCall}
            className={`
              relative group overflow-hidden px-10 py-4 rounded-full font-bold text-lg shadow-xl transition-all duration-300 transform hover:-translate-y-0.5 hover:shadow-2xl active:translate-y-0 active:shadow-md
              ${isConnected 
                ? 'bg-gray-900 text-white hover:bg-gray-800 ring-4 ring-gray-100' 
                : `${isChloe ? 'bg-emerald-600 hover:bg-emerald-500 ring-4 ring-emerald-100' : 'bg-red-600 hover:bg-red-500 ring-4 ring-red-100'} text-white`
              }
            `}
          >
            <div className="relative flex items-center gap-3 z-10">
              {isConnected ? <PhoneOff className="w-6 h-6" /> : <Phone className="w-6 h-6" />}
              <span>{isConnected ? 'End Session' : 'Start Call'}</span>
            </div>
          </button>
        </div>
      </div>

      <footer className="mt-8 text-center">
        <p className="text-xs font-semibold text-gray-400">Green Choice Heating & Cooling © 2026</p>
        <p className="text-[10px] text-gray-300 mt-1">Powered by Google Gemini Multimodal Live API</p>
      </footer>
    </div>
  );
}
