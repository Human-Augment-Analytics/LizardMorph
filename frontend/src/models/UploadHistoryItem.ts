export interface UploadHistoryItem {
  name: string;
  timestamp: string;
  index: number;
  viewType?: string; // Store the view type to filter history by class
}
