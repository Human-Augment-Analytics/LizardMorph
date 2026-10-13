import { Component } from "react";
import { getHistoryPanelStyles } from "./HistoryPanel.style";
import type { UploadHistoryItem } from "../models/UploadHistoryItem";
import type { ResolvedTheme } from "../contexts/theme";

interface HistoryPanelProps {
  uploadHistory: UploadHistoryItem[];
  currentImageIndex: number;
  uploadProgress: { [key: string]: number };
  onSelectImage: (index: number) => void;
  onLoadFromUploads: (filename: string) => void;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
  theme: ResolvedTheme;
}

interface HistoryPanelState {
  isCollapsedHovered: boolean;
  isCollapsedFocused: boolean;
  isCollapseBtnHovered: boolean;
  isCollapseBtnFocused: boolean;
  focusedRowIndex: number | null;
}

export class HistoryPanel extends Component<HistoryPanelProps, HistoryPanelState> {
  state: HistoryPanelState = {
    isCollapsedHovered: false,
    isCollapsedFocused: false,
    isCollapseBtnHovered: false,
    isCollapseBtnFocused: false,
    focusedRowIndex: null,
  };

  render() {
    const {
      uploadHistory,
      currentImageIndex,
      uploadProgress,
      onSelectImage,
      onLoadFromUploads,
      isCollapsed,
      onToggleCollapse,
      theme,
    } = this.props;
    const styles = getHistoryPanelStyles(theme);

    if (isCollapsed) {
      return (
        <div
          style={{
            ...styles.collapsedContainer,
            ...(this.state.isCollapsedHovered || this.state.isCollapsedFocused
              ? styles.collapsedContainerHovered
              : {}),
          }}
          onClick={onToggleCollapse}
          onMouseEnter={() => this.setState({ isCollapsedHovered: true })}
          onMouseLeave={() => this.setState({ isCollapsedHovered: false })}
          title="Show history"
          data-testid="history-panel-collapsed"
        >
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onToggleCollapse?.();
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                e.stopPropagation();
                onToggleCollapse?.();
              }
            }}
            onFocus={() => this.setState({ isCollapsedFocused: true })}
            onBlur={() => this.setState({ isCollapsedFocused: false })}
            style={{
              ...styles.collapsedButton,
              ...(this.state.isCollapsedFocused ? styles.focusRing : {}),
            }}
            aria-label="Show history panel"
            aria-expanded={false}
            title="Show history"
          >
            <span style={styles.collapsedIcon} aria-hidden="true">▶</span>
            <span style={styles.collapsedText}>📂 History</span>
          </button>
        </div>
      );
    }

    return (
      <div style={styles.historyContainer}>
        <div style={styles.historyHeader}>
          <h3 style={styles.historyTitle}>History</h3>
          {onToggleCollapse && (
            <button
              type="button"
              onClick={onToggleCollapse}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onToggleCollapse?.();
                }
              }}
              onMouseEnter={() => this.setState({ isCollapseBtnHovered: true })}
              onMouseLeave={() => this.setState({ isCollapseBtnHovered: false })}
              onFocus={() => this.setState({ isCollapseBtnFocused: true })}
              onBlur={() => this.setState({ isCollapseBtnFocused: false })}
              style={{
                ...styles.collapseButton,
                ...(this.state.isCollapseBtnHovered || this.state.isCollapseBtnFocused
                  ? styles.collapseButtonHovered
                  : {}),
                ...(this.state.isCollapseBtnFocused ? styles.focusRing : {}),
              }}
              aria-label="Hide history panel"
              aria-expanded={true}
              title="Hide history panel"
            >
              ◀ Hide
            </button>
          )}
        </div>
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
                      tabIndex={0}
                      role="button"
                      aria-label={`Select ${item.name}`}
                      aria-selected={item.index === currentImageIndex}
                      onClick={() =>
                        item.index >= 0
                          ? onSelectImage(item.index)
                          : onLoadFromUploads(item.name)
                      }
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          if (item.index >= 0) {
                            onSelectImage(item.index);
                          } else {
                            onLoadFromUploads(item.name);
                          }
                        }
                      }}
                      onFocus={() => this.setState({ focusedRowIndex: idx })}
                      onBlur={() => this.setState({ focusedRowIndex: null })}
                      style={{
                        ...styles.historyTableRow,
                        ...(item.index === currentImageIndex
                          ? styles.historyTableRowSelected
                          : {}),
                        ...(this.state.focusedRowIndex === idx ? styles.focusRing : {}),
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
                        <div style={styles.historyItemTimestamp}>
                          {item.timestamp}
                        </div>
                        {isUploading && (
                          <div style={styles.progressContainer}>
                            {progress === -1 ? (
                              <div style={styles.progressError}>
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
