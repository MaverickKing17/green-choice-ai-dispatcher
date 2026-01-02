export enum AgentPersona {
  CHLOE = 'CHLOE',
  SAM = 'SAM'
}

export interface MessageLog {
  role: 'user' | 'model' | 'system';
  text: string;
  timestamp: Date;
}

export interface AudioVisualizerProps {
  isActive: boolean;
  volume: number;
  persona: AgentPersona;
}

export interface BlobData {
    data: string;
    mimeType: string;
}
