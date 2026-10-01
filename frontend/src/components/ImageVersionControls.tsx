import { Component } from "react";
import { getImageVersionControlsStyles } from "./ImageVersionControls.style";
import type { ImageSet } from "../models/ImageSet";
import type { ResolvedTheme } from "../contexts/theme";

interface ImageVersionControlsProps {
  dataFetched: boolean;
  imageSet: ImageSet;
  currentImageURL: string | null;
  loading: boolean;
  dataLoading: boolean;
  onVersionChange: (url: string) => void;
  isEditMode: boolean;
  onToggleEditMode: () => void;
  onResetZoom: () => void;
  /** Toepad view only: swaps the full image for a 2x2 grid of toepad close-ups */
  isToepadGridOn?: boolean;
  onToggleToepadGrid?: () => void;
  theme: ResolvedTheme;
}

export class ImageVersionControls extends Component<ImageVersionControlsProps> {
  render() {
    const {
      dataFetched,
      imageSet,
      currentImageURL,
      loading,
      dataLoading,
      onVersionChange,
      isEditMode,
      onToggleEditMode,
      onResetZoom,
      isToepadGridOn,
      onToggleToepadGrid,
      theme,
    } = this.props;
    const styles = getImageVersionControlsStyles(theme);

    if (!dataFetched || !imageSet.original) return null;
    return (
      <div style={{
        ...styles.imageVersionButtons,
      }}>
        <button
          onClick={() => onVersionChange(imageSet.original)}
          disabled={loading || dataLoading}
          style={{
            ...styles.versionButton,
            ...(currentImageURL === imageSet.original
              ? styles.versionButtonActive
              : {}),
            ...(loading || dataLoading
              ? styles.versionButtonDisabled
              : {}),
          }}
        >
          Original
        </button>
        <button
          onClick={() => onVersionChange(imageSet.inverted)}
          disabled={loading || dataLoading}
          style={{
            ...styles.versionButton,
            ...(currentImageURL === imageSet.inverted
              ? styles.versionButtonActive
              : {}),
            ...(loading || dataLoading
              ? styles.versionButtonDisabled
              : {}),
          }}
        >
          Inverted
        </button>
        <button
          onClick={() => onVersionChange(imageSet.color_contrasted)}
          disabled={loading || dataLoading}
          style={{
            ...styles.versionButton,
            ...(currentImageURL === imageSet.color_contrasted
              ? styles.versionButtonActive
              : {}),
            ...(loading || dataLoading
              ? styles.versionButtonDisabled
              : {}),
          }}
        >
          Color Contrasted
        </button>
        {/* Edit Points and Reset Zoom buttons */}
        <button
          onClick={onToggleEditMode}
          style={{
            padding: '8px 16px',
            backgroundColor: isEditMode ? '#ffc107' : '#4F7942',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: 'bold',
            marginLeft: '16px',
          }}
        >
          {isEditMode ? 'Save Points' : 'Edit Points'}
        </button>
        <button
          onClick={onResetZoom}
          style={{
            padding: '8px 16px',
            backgroundColor: '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: 'bold',
          }}
        >
          Reset Zoom
        </button>
        {onToggleToepadGrid && (
          <button
            onClick={onToggleToepadGrid}
            aria-pressed={Boolean(isToepadGridOn)}
            title="Show all four toepads in a 2x2 grid"
            style={{
              ...styles.versionButton,
              ...(isToepadGridOn ? styles.versionButtonActive : {}),
              fontWeight: 'bold',
              marginLeft: '16px',
            }}
          >
            {`Toepad Grid: ${isToepadGridOn ? 'On' : 'Off'}`}
          </button>
        )}
      </div>
    );
  }
}
