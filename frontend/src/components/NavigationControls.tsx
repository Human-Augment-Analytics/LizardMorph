import { Component } from "react";
import { getNavigationControlsStyles } from "./NavigationControls.style";
import type { ResolvedTheme } from "../contexts/theme";
import type { ReviewStatus } from "../models/UploadHistoryItem";

interface NavigationControlsProps {
  currentImageIndex: number;
  totalImages: number;
  loading: boolean;
  onPrevious: () => void;
  onNext: () => void;
  currentReviewStatus?: ReviewStatus;
  onApproveAndNext?: () => void;
  onChangeReviewStatus?: (status: ReviewStatus) => void;
  theme: ResolvedTheme;
}

export class NavigationControls extends Component<NavigationControlsProps> {
  render() {
    const {
      currentImageIndex,
      totalImages,
      loading,
      onPrevious,
      onNext,
      currentReviewStatus,
      onApproveAndNext,
      onChangeReviewStatus,
      theme,
    } = this.props;
    const styles = getNavigationControlsStyles(theme);

    if (totalImages <= 1 && !onApproveAndNext && !onChangeReviewStatus) {
      if (totalImages === 0) return null;
    }

    const resolvedStatus: ReviewStatus = currentReviewStatus || "unreviewed";

    return (
      <div style={styles.navigationControls}>
        {totalImages > 1 && (
          <button
            type="button"
            onClick={onPrevious}
            disabled={currentImageIndex === 0 || loading}
            style={{
              ...styles.navButton,
              ...(currentImageIndex === 0 || loading
                ? styles.navButtonDisabled
                : {}),
            }}
          >
            Previous Image
          </button>
        )}

        {totalImages > 0 && (
          <span style={styles.imageCounter}>
            Image {currentImageIndex + 1} of {totalImages}
          </span>
        )}

        {totalImages > 1 && (
          <button
            type="button"
            onClick={onNext}
            disabled={currentImageIndex === totalImages - 1 || loading}
            style={{
              ...styles.navButton,
              ...(currentImageIndex === totalImages - 1 || loading
                ? styles.navButtonDisabled
                : {}),
            }}
          >
            Next Image
          </button>
        )}

        {(onChangeReviewStatus || onApproveAndNext) && totalImages > 0 && (
          <span style={styles.divider} aria-hidden="true" />
        )}

        {/* Review / QC Status Dropdown */}
        {onChangeReviewStatus && totalImages > 0 && (
          <div style={styles.statusSelector} title="Change specimen review status">
            <span>Status:</span>
            <select
              data-testid="qc-status-select"
              aria-label="Specimen review status"
              value={resolvedStatus}
              onChange={(e) =>
                onChangeReviewStatus(e.target.value as ReviewStatus)
              }
              style={styles.statusSelectDropdown}
            >
              <option value="unreviewed">⚪ Unreviewed</option>
              <option value="needs_review">🟡 Needs Review</option>
              <option value="verified">🟢 Verified</option>
            </select>
          </div>
        )}

        {/* One-click "Approve & Next Image" button */}
        {onApproveAndNext && totalImages > 0 && (
          <button
            type="button"
            data-testid="approve-and-next-button"
            onClick={onApproveAndNext}
            disabled={loading}
            title="Mark specimen as Verified and proceed to next image (Space / Enter)"
            style={{
              ...styles.approveButton,
              ...(loading ? styles.approveButtonDisabled : {}),
            }}
          >
            <span>✓ Approve &amp; Next</span>
            <kbd style={styles.approveKbd}>Space</kbd>
          </button>
        )}
      </div>
    );
  }
}
