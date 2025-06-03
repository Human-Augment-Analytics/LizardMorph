import { Component } from "react";
import { HistoryPanelStyles } from "./HistoryPanel.style";
import type { UploadHistoryItem } from "../models/UploadHistoryItem";

interface HistoryPanelProps {
  uploadHistory: UploadHistoryItem[];
  currentImageIndex: number;
  onSelectImage: (index: number) => void;
  onLoadFromUploads: (filename: string) => void;
}

export class HistoryPanel extends Component<HistoryPanelProps> {
  render() {
    const {
      uploadHistory,
      currentImageIndex,
      onSelectImage,
      onLoadFromUploads,
    } = this.props;
    return (
      <div style={HistoryPanelStyles.historyContainer}>
        <h3>History</h3>
        <div style={HistoryPanelStyles.historyTableContainer}>
          <table style={HistoryPanelStyles.historyTable}>
            <thead>
              <tr style={HistoryPanelStyles.historyTableHeader}>
                <th style={HistoryPanelStyles.historyTableHeaderCell}>
                  Image Name
                </th>
              </tr>
            </thead>
            <tbody>
              {uploadHistory.length > 0 ? (
                uploadHistory.map((item, idx) => (
                  <tr
                    key={`${item.name}-${idx}`}
                    onClick={() =>
                      item.index >= 0
                        ? onSelectImage(item.index)
                        : onLoadFromUploads(item.name)
                    }
                    style={{
                      ...HistoryPanelStyles.historyTableRow,
                      ...(item.index === currentImageIndex
                        ? HistoryPanelStyles.historyTableRowSelected
                        : {}),
                    }}
                  >
                    <td
                      style={{
                        ...HistoryPanelStyles.historyTableCell,
                        ...(item.index === currentImageIndex
                          ? HistoryPanelStyles.historyTableCellSelected
                          : {}),
                      }}
                    >
                      {item.name}
                      <div style={{ fontSize: "0.8em", color: "#666" }}>
                        {item.timestamp}
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td style={HistoryPanelStyles.historyTableEmptyCell}>
                    No images in history
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    );
  }
}
