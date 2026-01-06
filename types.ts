
export enum AgentPersona {
  CHLOE = 'CHLOE',
  MIKE = 'MIKE'
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
