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
    } = this.props;

    if (!dataFetched || !imageSet.original) return null;
    return (
      <div style={ImageVersionControlsStyles.imageVersionButtons}>
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
      </div>
    );
  }
}
