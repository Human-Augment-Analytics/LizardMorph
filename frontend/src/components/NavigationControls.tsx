import { Component } from "react";
import { NavigationControlsStyles } from "./NavigationControls.style";

interface NavigationControlsProps {
  currentImageIndex: number;
  totalImages: number;
  loading: boolean;
  onPrevious: () => void;
  onNext: () => void;
}

export class NavigationControls extends Component<NavigationControlsProps> {
  render() {
    const { currentImageIndex, totalImages, loading, onPrevious, onNext } =
      this.props;

    if (totalImages <= 1) return null;
    return (
      <div style={NavigationControlsStyles.navigationControls}>
        <button
          onClick={onPrevious}
          disabled={currentImageIndex === 0 || loading}
          style={{
            ...NavigationControlsStyles.navButton,
            ...(currentImageIndex === 0 || loading
              ? NavigationControlsStyles.navButtonDisabled
              : {}),
          }}
        >
          Previous Image
        </button>

        <span style={NavigationControlsStyles.imageCounter}>
          Image {currentImageIndex + 1} of {totalImages}
        </span>

        <button
          onClick={onNext}
          disabled={currentImageIndex === totalImages - 1 || loading}
          style={{
            ...NavigationControlsStyles.navButton,
            ...(currentImageIndex === totalImages - 1 || loading
              ? NavigationControlsStyles.navButtonDisabled
              : {}),
          }}
        >
          Next Image
        </button>
      </div>
    );
  }
}
