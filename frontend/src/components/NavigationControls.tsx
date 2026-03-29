import { Component } from "react";
import { getNavigationControlsStyles } from "./NavigationControls.style";
import type { ResolvedTheme } from "../contexts/ThemeContext";

interface NavigationControlsProps {
  currentImageIndex: number;
  totalImages: number;
  loading: boolean;
  onPrevious: () => void;
  onNext: () => void;
  theme: ResolvedTheme;
}

export class NavigationControls extends Component<NavigationControlsProps> {
  render() {
    const { currentImageIndex, totalImages, loading, onPrevious, onNext, theme } =
      this.props;
    const styles = getNavigationControlsStyles(theme);

    if (totalImages <= 1) return null;
    return (
      <div style={styles.navigationControls}>
        <button
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

        <span style={styles.imageCounter}>
          Image {currentImageIndex + 1} of {totalImages}
        </span>

        <button
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
      </div>
    );
  }
}
