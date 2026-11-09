import { Component } from "react";
import { getHistoryPanelStyles } from "./HistoryPanel.style";
import type { UploadHistoryItem, ReviewStatus } from "../models/UploadHistoryItem";
import type { ResolvedTheme } from "../contexts/theme";

interface HistoryPanelProps {
  uploadHistory: UploadHistoryItem[];
  currentImageIndex: number;
  uploadProgress: { [key: string]: number };
  onSelectImage: (index: number) => void;
  onLoadFromUploads: (filename: string) => void;
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
  onChangeReviewStatus?: (name: string, status: ReviewStatus) => void;
  theme: ResolvedTheme;
}

interface HistoryPanelState {
  isCollapsedHovered: boolean;
  isCollapsedFocused: boolean;
  isCollapseBtnHovered: boolean;
  isCollapseBtnFocused: boolean;
  focusedRowIndex: number | null;
  statusFilter: "all" | ReviewStatus;
}

export class HistoryPanel extends Component<HistoryPanelProps, HistoryPanelState> {
  state: HistoryPanelState = {
    isCollapsedHovered: false,
    isCollapsedFocused: false,
    isCollapseBtnHovered: false,
    isCollapseBtnFocused: false,
    focusedRowIndex: null,
    statusFilter: "all",
  };

  private renderStatusBadge(status?: ReviewStatus) {
    const { theme } = this.props;
    const styles = getHistoryPanelStyles(theme);
    const resolvedStatus = status || "unreviewed";

    let label = "Unreviewed";
    let icon = "⚪";
    let badgeStyle = styles.statusBadgeUnreviewed;
    let title = "Unreviewed specimen";

    if (resolvedStatus === "verified") {
      label = "Verified";
      icon = "🟢";
      badgeStyle = styles.statusBadgeVerified;
      title = "Verified by biologist";
    } else if (resolvedStatus === "needs_review") {
      label = "Needs Review";
      icon = "🟡";
      badgeStyle = styles.statusBadgeNeedsReview;
      title = "Needs review (low confidence or missing toe)";
    }

    return (
      <span
        data-testid={`status-badge-${resolvedStatus}`}
        style={{ ...styles.statusBadge, ...badgeStyle }}
        title={title}
      >
        <span aria-hidden="true">{icon}</span>
        <span>{label}</span>
      </span>
    );
  }

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
    const filter = this.state.statusFilter;

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

    const unreviewedCount = uploadHistory.filter(
      (i) => !i.reviewStatus || i.reviewStatus === "unreviewed"
    ).length;
    const needsReviewCount = uploadHistory.filter(
      (i) => i.reviewStatus === "needs_review"
    ).length;
    const verifiedCount = uploadHistory.filter(
      (i) => i.reviewStatus === "verified"
    ).length;

    const filteredHistory = uploadHistory.filter((item) => {
      if (filter === "all") return true;
      const itemStatus = item.reviewStatus || "unreviewed";
      return itemStatus === filter;
    });

    return (
      <div style={styles.historyContainer}>
        <div style={styles.historyHeader}>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px", width: "100%" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
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
            {uploadHistory.length > 0 && (
              <div
                style={styles.filterToolbar}
                role="tablist"
                aria-label="Review status filters"
              >
                <button
                  type="button"
                  data-testid="filter-all"
                  onClick={() => this.setState({ statusFilter: "all" })}
                  style={{
                    ...styles.filterButton,
                    ...(filter === "all" ? styles.filterButtonActive : {}),
                  }}
                >
                  All ({uploadHistory.length})
                </button>
                <button
                  type="button"
                  data-testid="filter-needs-review"
                  onClick={() => this.setState({ statusFilter: "needs_review" })}
                  style={{
                    ...styles.filterButton,
                    ...(filter === "needs_review" ? styles.filterButtonActive : {}),
                  }}
                  title="Filter by Needs Review"
                >
                  🟡 Needs Review ({needsReviewCount})
                </button>
                <button
                  type="button"
                  data-testid="filter-verified"
                  onClick={() => this.setState({ statusFilter: "verified" })}
                  style={{
                    ...styles.filterButton,
                    ...(filter === "verified" ? styles.filterButtonActive : {}),
                  }}
                  title="Filter by Verified"
                >
                  🟢 Verified ({verifiedCount})
                </button>
                <button
                  type="button"
                  data-testid="filter-unreviewed"
                  onClick={() => this.setState({ statusFilter: "unreviewed" })}
                  style={{
                    ...styles.filterButton,
                    ...(filter === "unreviewed" ? styles.filterButtonActive : {}),
                  }}
                  title="Filter by Unreviewed"
                >
                  ⚪ Unreviewed ({unreviewedCount})
                </button>
              </div>
            )}
          </div>
        </div>
        <div style={styles.historyTableContainer}>
          <table style={styles.historyTable}>
            <thead>
              <tr style={styles.historyTableHeader}>
                <th style={styles.historyTableHeaderCell}>
                  Image Name & Status
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredHistory.length > 0 ? (
                filteredHistory.map((item, idx) => {
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
                        <div style={styles.historyItemNameRow}>
                          <span style={styles.historyItemName} title={item.name}>
                            {item.name}
                          </span>
                          {this.renderStatusBadge(item.reviewStatus)}
                        </div>
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
                    {uploadHistory.length === 0
                      ? "No images in history"
                      : `No images matching "${filter}"`}
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
