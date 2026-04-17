import { Component } from "react";
import { getHistoryPanelStyles } from "./HistoryPanel.style";
import type { UploadHistoryItem } from "../models/UploadHistoryItem";
import type { ResolvedTheme } from "../contexts/ThemeContext";

interface HistoryPanelProps {
  uploadHistory: UploadHistoryItem[];
  currentImageIndex: number;
  uploadProgress: { [key: string]: number };
  onSelectImage: (index: number) => void;
  onLoadFromUploads: (filename: string) => void;
  theme: ResolvedTheme;
}

export class HistoryPanel extends Component<HistoryPanelProps> {
  render() {
    const {
      uploadHistory,
      currentImageIndex,
      uploadProgress,
      onSelectImage,
      onLoadFromUploads,
      theme,
    } = this.props;
    const styles = getHistoryPanelStyles(theme);
    return (
      <div style={styles.historyContainer}>
        <h3>History</h3>
        <div style={styles.historyTableContainer}>
          <table style={styles.historyTable}>
            <thead>
              <tr style={styles.historyTableHeader}>
                <th style={styles.historyTableHeaderCell}>
                  Image Name
                </th>
              </tr>
            </thead>
            <tbody>
              {uploadHistory.length > 0 ? (
                uploadHistory.map((item, idx) => {
                  const isUploading = uploadProgress[item.name] !== undefined;
                  const progress = uploadProgress[item.name] || 0;

                  return (
                    <tr
                      key={`${item.name}-${idx}`}
                      onClick={() =>
                        item.index >= 0
                          ? onSelectImage(item.index)
                          : onLoadFromUploads(item.name)
                      }
                      style={{
                        ...styles.historyTableRow,
                        ...(item.index === currentImageIndex
                          ? styles.historyTableRowSelected
                          : {}),
                      }}
                    >
                      <td
                        style={{
                          ...styles.historyTableCell,
                          ...(item.index === currentImageIndex
                            ? styles.historyTableCellSelected
                            : {}),
                        }}
                      >
                        {item.name}
                        <div style={{ fontSize: "0.8em", color: theme === "dark" ? "#8899aa" : "#666" }}>
                          {item.timestamp}
                        </div>
                        {isUploading && (
                          <div style={styles.progressContainer}>
                            {progress === -1 ? (
                              <div style={{ color: "red", fontSize: "0.8em" }}>
                                Error processing image
                              </div>
                            ) : (
                              <>
                                <div style={styles.progressBar}>
                                  <div
                                    style={{
                                      ...styles.progressFill,
                                      width: `${progress}%`,
                                    }}
                                  />
                                </div>
                                <span style={styles.progressText}>
                                  {progress}%
                                </span>
                              </>
                            )}
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td style={styles.historyTableEmptyCell}>
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
