import React, { useState, useRef, useEffect, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type, FunctionDeclaration } from "@google/genai";
import { AgentPersona, MessageLog, BlobData } from './types';
import { Mic, MicOff, Phone, PhoneOff, AlertTriangle, Leaf, History, Activity } from 'lucide-react';

// --- Constants & Config ---
const API_KEY = process.env.API_KEY || ''; // Injected by environment
const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025';

// --- System Instructions ---
const SYSTEM_INSTRUCTION = `
You are the Voice AI system for 'Green Choice Heating & Cooling' in East York.
You have two distinct personas. You start as CHLOE.

PERSONA 1: CHLOE (Front-Desk)
- Tone: Warm, polite, enthusiastic, professional.
- Role: Rebate expert, general inquiries.
- Key Knowledge: 2026 Home Renovation Savings program (specifically the $7,500 electric-to-heat-pump tier).
- Task: Qualify leads for rebates.

PERSONA 2: SAM (Emergency Dispatcher)
- Tone: Calm, direct, fast, authoritative.
- Role: Emergency response.
- Triggers: "gas smell", "no heat", "water leak".
- Task: Promise 4-hour response, get address immediately.

CRITICAL LOGIC FLOW:
1. Start the conversation as CHLOE. Greet the customer warmly.
2. Listen carefully to the user.
3. IF the user mentions "gas smell", "no heat", or "water leak":
   a. You MUST say exactly: "One moment, I'm handing you over to Sam, our emergency specialist."
   b. Call the tool function \`switchToSam\`.
   c. IMMEDIATELY switch your tone and persona to SAM for the rest of the conversation. Do not switch back.
4. IF it is a routine booking/inquiry (CHLOE):
   a. Collect name and phone number.
   b. If booking is complete, call the tool function \`startSurvey\`.
   c. Ask satisfaction questions.

Keep responses concise (under 30 seconds) and spoken naturally.
`;

// --- Tool Definitions ---
const switchToSamTool: FunctionDeclaration = {
  name: 'switchToSam',
  description: 'Triggers the visual and logical switch to Sam, the emergency dispatcher, when an emergency is detected.',
  parameters: {
    type: Type.OBJECT,
    properties: {},
  },
};

const startSurveyTool: FunctionDeclaration = {
  name: 'startSurvey',
  description: 'Triggers the satisfaction survey mode after a routine call is wrapped up.',
  parameters: {
    type: Type.OBJECT,
    properties: {},
  },
};

const tools = [{ functionDeclarations: [switchToSamTool, startSurveyTool] }];

// --- Audio Helpers ---

function createBlob(data: Float32Array): BlobData {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    // Clamp values to [-1, 1] before scaling
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

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
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

// --- Components ---

const Visualizer: React.FC<{ volume: number; persona: AgentPersona; isActive: boolean }> = ({ volume, persona, isActive }) => {
  const bars = 5;
  
  return (
    <div className="flex items-center justify-center space-x-2 h-16">
      {isActive ? Array.from({ length: bars }).map((_, i) => {
        // Simple visualizer logic based on volume
        const height = Math.max(10, Math.min(100, volume * (100 + i * 20))); 
        const colorClass = persona === AgentPersona.CHLOE ? 'bg-green-500' : 'bg-red-600';
        
        return (
          <div
            key={i}
            className={`w-3 rounded-full transition-all duration-100 ${colorClass}`}
            style={{ height: `${height}%` }}
          />
        );
      }) : (
        <div className="text-gray-400 text-sm font-medium">Ready to connect...</div>
      )}
    </div>
  );
};

const TranscriptMessage: React.FC<{ msg: MessageLog }> = ({ msg }) => {
  const isUser = msg.role === 'user';
  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-3`}>
      <div className={`max-w-[80%] rounded-lg px-4 py-2 text-sm ${
        isUser 
          ? 'bg-blue-600 text-white rounded-br-none' 
          : 'bg-white border border-gray-200 text-gray-800 rounded-bl-none shadow-sm'
      }`}>
        {msg.text}
      </div>
    </div>
  );
};

// --- Main App ---

export default function App() {
  // State
  const [isConnected, setIsConnected] = useState(false);
  const [currentPersona, setCurrentPersona] = useState<AgentPersona>(AgentPersona.CHLOE);
  const [volumeLevel, setVolumeLevel] = useState(0);
  const [transcripts, setTranscripts] = useState<MessageLog[]>([]);
  const [isSurveyMode, setIsSurveyMode] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Refs for Audio/Gemini logic
  const nextStartTime = useRef<number>(0);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const inputSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sessionRef = useRef<any>(null); // To hold the active session
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  // Initialization
  useEffect(() => {
    // Cleanup on unmount
    return () => {
      disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const disconnect = useCallback(() => {
    if (sessionRef.current) {
      // There isn't a strictly documented close() on the session object in the preview, 
      // but we stop processing.
      sessionRef.current = null;
    }

    // Stop audio contexts
    if (inputAudioContextRef.current) {
      inputAudioContextRef.current.close();
      inputAudioContextRef.current = null;
    }
    if (outputAudioContextRef.current) {
      outputAudioContextRef.current.close();
      outputAudioContextRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (inputSourceRef.current) {
      inputSourceRef.current.disconnect();
      inputSourceRef.current = null;
    }
    
    // Stop all playing sources
    sourcesRef.current.forEach(source => {
      try { source.stop(); } catch(e) {}
    });
    sourcesRef.current.clear();

    setIsConnected(false);
    setVolumeLevel(0);
  }, []);

  const connect = async () => {
    setErrorMsg(null);
    try {
      // 1. Initialize Audio Contexts
      const InputContextClass = (window.AudioContext || (window as any).webkitAudioContext);
      const inputCtx = new InputContextClass({ sampleRate: 16000 });
      const outputCtx = new InputContextClass({ sampleRate: 24000 });
      
      inputAudioContextRef.current = inputCtx;
      outputAudioContextRef.current = outputCtx;
      
      const outputNode = outputCtx.createGain();
      outputNode.connect(outputCtx.destination);

      // 2. Get Microphone Stream
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // 3. Initialize Gemini Client
      const ai = new GoogleGenAI({ apiKey: API_KEY });

      // 4. Setup Connection Promise
      // We use a promise wrapper to ensure we have the session before sending data
      let resolveSession: (s: any) => void;
      const sessionPromise = new Promise<any>(resolve => {
        resolveSession = resolve;
      });

      // 5. Connect to Live API
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
            console.log('Gemini Live Session Opened');
            resolveSession(session); // Resolve our internal promise
            setIsConnected(true);
            setCurrentPersona(AgentPersona.CHLOE); // Reset to Chloe on new call
            setIsSurveyMode(false);
            setTranscripts([]);

            // Start Audio Input Pipeline
            const source = inputCtx.createMediaStreamSource(stream);
            inputSourceRef.current = source;
            
            // ScriptProcessor is deprecated but easiest for raw PCM in vanilla React without Worklet file loading complexity
            const processor = inputCtx.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;

            processor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              
              // Calculate volume for visualizer
              let sum = 0;
              for(let i=0; i<inputData.length; i++) sum += inputData[i] * inputData[i];
              const rms = Math.sqrt(sum / inputData.length);
              setVolumeLevel(prev => Math.max(rms * 5, prev * 0.9)); // Smooth decay

              const pcmBlob = createBlob(inputData);
              
              // Send to Gemini
              sessionPromise.then(sess => {
                sess.sendRealtimeInput({ media: pcmBlob });
              });
            };

            source.connect(processor);
            processor.connect(inputCtx.destination);
          },
          onmessage: async (message: LiveServerMessage) => {
            // Handle Tool Calls (The Brains of the Persona Switch)
            if (message.toolCall) {
              for (const fc of message.toolCall.functionCalls) {
                console.log('Tool Call:', fc.name);
                
                if (fc.name === 'switchToSam') {
                  setCurrentPersona(AgentPersona.SAM);
                  setTranscripts(prev => [...prev, {
                    role: 'system',
                    text: '⚠️ EMERGENCY DETECTED: Handoff to Sam initiated.',
                    timestamp: new Date()
                  }]);
                } else if (fc.name === 'startSurvey') {
                  setIsSurveyMode(true);
                  setTranscripts(prev => [...prev, {
                    role: 'system',
                    text: '📝 Routine call complete. Starting survey.',
                    timestamp: new Date()
                  }]);
                }

                // Send response back to acknowledge tool execution
                sessionPromise.then(sess => {
                  sess.sendToolResponse({
                    functionResponses: {
                      id: fc.id,
                      name: fc.name,
                      response: { result: "ok" }
                    }
                  });
                });
              }
            }

            // Handle Transcripts (for UI log)
            if (message.serverContent?.inputTranscription) {
               const text = message.serverContent.inputTranscription.text;
               if (text) {
                 setTranscripts(prev => {
                   const last = prev[prev.length - 1];
                   if (last && last.role === 'user') {
                     // Simple debounce/append for streaming transcripts
                     return [...prev.slice(0, -1), { ...last, text: last.text + text }];
                   }
                   return [...prev, { role: 'user', text, timestamp: new Date() }];
                 });
               }
            }
            if (message.serverContent?.modelTurn?.parts?.[0]?.text) {
               // Model text output (usually hidden if audio is on, but useful for logs)
            }


            // Handle Audio Output
            const base64Audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (base64Audio && outputAudioContextRef.current) {
              const ctx = outputAudioContextRef.current;
              nextStartTime.current = Math.max(nextStartTime.current, ctx.currentTime);
              
              const audioBuffer = await decodeAudioData(
                decodeAudio(base64Audio),
                ctx,
                24000,
                1
              );
              
              const source = ctx.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(outputNode);
              
              source.addEventListener('ended', () => {
                sourcesRef.current.delete(source);
              });
              
              source.start(nextStartTime.current);
              nextStartTime.current += audioBuffer.duration;
              sourcesRef.current.add(source);
            }
            
            // Handle Interruption
            if (message.serverContent?.interrupted) {
              sourcesRef.current.forEach(s => s.stop());
              sourcesRef.current.clear();
              nextStartTime.current = 0;
            }
          },
          onclose: () => {
            console.log("Session Closed");
            setIsConnected(false);
          },
          onerror: (err) => {
            console.error("Session Error", err);
            setErrorMsg("Connection error. Please try again.");
            disconnect();
          }
        }
      });
      
      sessionRef.current = session;

    } catch (e) {
      console.error(e);
      setErrorMsg("Failed to access microphone or connect to AI.");
      disconnect();
    }
  };

  const handleToggleCall = () => {
    if (isConnected) {
      disconnect();
    } else {
      connect();
    }
  };

  // --- Render Helpers ---
  const isChloe = currentPersona === AgentPersona.CHLOE;
  const themeColor = isChloe ? 'green' : 'red';
  const bgColor = isChloe ? 'bg-emerald-50' : 'bg-red-50';
  const borderColor = isChloe ? 'border-emerald-200' : 'border-red-200';
  const buttonColor = isChloe ? 'bg-emerald-600 hover:bg-emerald-700' : 'bg-red-600 hover:bg-red-700';

  return (
    <div className={`min-h-screen ${bgColor} transition-colors duration-700 flex flex-col items-center p-6`}>
      
      {/* Header / Nav */}
      <header className="w-full max-w-md flex items-center justify-between mb-8">
        <div className="flex items-center space-x-2">
          <Leaf className={`w-6 h-6 ${isChloe ? 'text-emerald-600' : 'text-gray-400'}`} />
          <h1 className="text-xl font-bold text-gray-800">Green Choice</h1>
        </div>
        <div className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          {isConnected ? 'Live Connection' : 'Offline'}
        </div>
      </header>

      {/* Main Agent Card */}
      <div className={`w-full max-w-md bg-white rounded-3xl shadow-xl overflow-hidden border-2 ${borderColor} transition-all duration-500`}>
        
        {/* Agent Persona Visual */}
        <div className={`relative h-64 w-full flex flex-col items-center justify-center p-6 transition-colors duration-500 ${isChloe ? 'bg-emerald-100' : 'bg-red-100'}`}>
          <div className="absolute top-4 right-4">
             {isSurveyMode && <span className="px-2 py-1 bg-white/50 rounded text-xs font-bold text-gray-600">SURVEY MODE</span>}
          </div>

          {/* Avatar Ring */}
          <div className={`relative w-32 h-32 rounded-full border-4 ${isChloe ? 'border-white' : 'border-red-500'} shadow-lg mb-4 flex items-center justify-center overflow-hidden bg-white`}>
            {isConnected && (
              <div className={`absolute inset-0 ${isChloe ? 'bg-emerald-400' : 'bg-red-500'} opacity-20 animate-audio-pulse`} style={{ animationDuration: `${Math.max(0.5, 2 - volumeLevel * 5)}s` }}></div>
            )}
            <img 
              src={isChloe 
                ? "https://picsum.photos/id/64/200/200" // Gentle portrait
                : "https://picsum.photos/id/1005/200/200" // More intense/focused portrait
              } 
              alt="Agent" 
              className="w-full h-full object-cover z-10"
            />
          </div>

          <h2 className="text-2xl font-bold text-gray-800 transition-all duration-300">
            {isChloe ? 'Chloe' : 'Sam'}
          </h2>
          <p className={`text-sm font-medium ${isChloe ? 'text-emerald-700' : 'text-red-700'}`}>
            {isChloe ? 'Front Desk & Rebate Specialist' : '⚠️ Emergency Dispatcher'}
          </p>
        </div>

        {/* Visualizer Area */}
        <div className="bg-gray-50 h-24 flex flex-col items-center justify-center border-b border-gray-100 relative">
          <Visualizer volume={volumeLevel} persona={currentPersona} isActive={isConnected} />
          {errorMsg && <p className="text-red-500 text-xs absolute bottom-1">{errorMsg}</p>}
        </div>

        {/* Controls */}
        <div className="p-6 flex justify-center">
          <button
            onClick={handleToggleCall}
            className={`flex items-center space-x-3 px-8 py-4 rounded-full shadow-lg text-white font-semibold text-lg transition-transform transform active:scale-95 ${isConnected ? 'bg-gray-800 hover:bg-gray-900' : buttonColor}`}
          >
            {isConnected ? (
              <>
                <PhoneOff className="w-6 h-6" />
                <span>End Call</span>
              </>
            ) : (
              <>
                <Phone className="w-6 h-6" />
                <span>Talk to Us</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Transcript Log (Simulating the backend integration or debug view) */}
      <div className="w-full max-w-md mt-8">
        <div className="flex items-center space-x-2 mb-4 text-gray-500">
           <History className="w-4 h-4" />
           <span className="text-sm font-medium">Live Transcript</span>
        </div>
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4 h-64 overflow-y-auto space-y-2">
          {transcripts.length === 0 && (
            <div className="text-center text-gray-400 text-sm mt-10">
              <Activity className="w-8 h-8 mx-auto mb-2 opacity-30" />
              Call history will appear here.
            </div>
          )}
          {transcripts.map((msg, idx) => (
             msg.role === 'system' ? (
                <div key={idx} className="flex justify-center my-2">
                  <span className="text-xs font-bold text-gray-500 bg-gray-100 px-2 py-1 rounded border border-gray-200">
                    {msg.text}
                  </span>
                </div>
             ) : (
                <TranscriptMessage key={idx} msg={msg} />
             )
          ))}
        </div>
      </div>

      <div className="mt-8 text-center max-w-xs text-gray-400 text-xs">
        <p>Green Choice Heating & Cooling © 2026.</p>
        <p className="mt-1">Powered by Gemini Multimodal Live API.</p>
      </div>

    </div>
  );
}
