import { Component } from "react";
import { ImageVersionControlsStyles } from "./ImageVersionControls.style";
import type { ImageSet } from "../models/ImageSet";

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
    } = this.props;

    if (!dataFetched || !imageSet.original) return null;
    return (
      <div style={{
        ...ImageVersionControlsStyles.imageVersionButtons,
      }}>
        <button
          onClick={() => onVersionChange(imageSet.original)}
          disabled={loading || dataLoading}
          style={{
            ...ImageVersionControlsStyles.versionButton,
            ...(currentImageURL === imageSet.original
              ? ImageVersionControlsStyles.versionButtonActive
              : {}),
            ...(loading || dataLoading
              ? ImageVersionControlsStyles.versionButtonDisabled
              : {}),
          }}
        >
          Original
        </button>
        <button
          onClick={() => onVersionChange(imageSet.inverted)}
          disabled={loading || dataLoading}
          style={{
            ...ImageVersionControlsStyles.versionButton,
            ...(currentImageURL === imageSet.inverted
              ? ImageVersionControlsStyles.versionButtonActive
              : {}),
            ...(loading || dataLoading
              ? ImageVersionControlsStyles.versionButtonDisabled
              : {}),
          }}
        >
          Inverted
        </button>
        <button
          onClick={() => onVersionChange(imageSet.color_contrasted)}
          disabled={loading || dataLoading}
          style={{
            ...ImageVersionControlsStyles.versionButton,
            ...(currentImageURL === imageSet.color_contrasted
              ? ImageVersionControlsStyles.versionButtonActive
              : {}),
            ...(loading || dataLoading
              ? ImageVersionControlsStyles.versionButtonDisabled
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
            backgroundColor: isEditMode ? '#ff4444' : '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: 'bold',
            marginLeft: '16px',
          }}
        >
          {isEditMode ? 'Exit' : 'Edit Points'}
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
      </div>
    );
  }
}
