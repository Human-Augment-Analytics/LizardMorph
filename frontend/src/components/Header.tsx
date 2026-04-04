import React, { Component } from "react";
import { getHeaderStyles } from "./Header.style";
import type { ResolvedTheme } from "../contexts/ThemeContext";
import lizard_logo from "../../public/lizard.svg";

interface HeaderProps {
  lizardCount: number;
  loading: boolean;
  dataFetched: boolean;
  dataError: Error | null;
  selectedViewType: string;
  onUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onExportAll: () => void;
  onClearHistory: () => void;
  onBackToSelection: () => void;
  onOpenMeasurementsModal?: () => void;
  toepadPredictorType?: string;
  onToepadPredictorTypeChange?: (type: string) => void;
  theme: ResolvedTheme;
}

interface HeaderState {
  isMenuOpen: boolean;
}

export class Header extends Component<HeaderProps, HeaderState> {
  constructor(props: HeaderProps) {
    super(props);
    this.state = {
      isMenuOpen: false,
    };
  }

  toggleMenu = () => {
    this.setState((prevState) => ({
      isMenuOpen: !prevState.isMenuOpen,
    }));
  };

  render() {
    const {
      lizardCount,
      loading,
      dataFetched,
      dataError,
      onUpload,
      onExportAll,
      onClearHistory,
      onOpenMeasurementsModal,
      selectedViewType,
      theme,
    } = this.props;

    const styles = getHeaderStyles(theme);
    const { isMenuOpen } = this.state;
    const isFree = selectedViewType === "free";
    const uploadLabel = loading
      ? "Uploading..."
      : isFree
        ? "Upload Images"
        : "Upload X-Ray Images";

    return (
      <div style={styles.header}>
        <div style={styles.infoBox}>
          <div style={styles.infoBoxContent}>
            <p style={styles.infoBoxParagraph}>
              Made with ❤️ by the Human Augmented Analytics Group (HAAG)
            </p>
            <p style={styles.infoBoxParagraph}>
              In Partnership with Dr. Stroud
            </p>
            <p style={styles.infoBoxItalic}>
              Georgia Institute of Technology
            </p>
            <div style={styles.infoBoxLinkRow}>
              <a
                href="https://github.com/Human-Augment-Analytics/Lizard-CV-Web-App"
                target="_blank"
                rel="noopener noreferrer"
                style={styles.infoBoxLink}
              >
                View on GitHub
              </a>
              <span
                style={styles.appVersion}
                title={`Build ${__BUILD_VERSION__}`}
              >
                {__APP_VERSION__}
              </span>
            </div>
            <div style={styles.lizardCount}>
              <strong>Number of Lizards Analyzed: {lizardCount}</strong>
            </div>
          </div>
        </div>
        <div style={styles.mainContent}>
          <div style={styles.buttonContainer}>
            <label
              htmlFor="file-upload"
              style={{
                ...styles.uploadButton,
                ...(loading ? styles.uploadButtonDisabled : {}),
              }}
            >
              {uploadLabel}
            </label>
            <input
              id="file-upload"
              type="file"
              accept="image/*"
              onChange={onUpload}
              style={{ display: "none" }}
              multiple
              disabled={loading}
            />
            <button
              onClick={onExportAll}
              disabled={!dataFetched || loading}
              style={{
                ...styles.exportButton,
                ...(!dataFetched || loading
                  ? styles.exportButtonDisabled
                  : {}),
              }}
            >
              Export All Data
            </button>

            <div style={styles.dropdownContainer}>
              <button
                onClick={this.toggleMenu}
                style={styles.dropdownButton}
              >
                More...
              </button>
              {isMenuOpen && (
                <div style={styles.dropdownContent}>
                  <button
                    onClick={onClearHistory}
                    disabled={loading}
                    style={{
                      ...styles.clearHistoryButton,
                      ...(loading
                        ? styles.clearHistoryButtonDisabled
                        : {}),
                    }}
                  >
                    Clear History
                  </button>
                  {onOpenMeasurementsModal && (
                    <button
                      onClick={onOpenMeasurementsModal}
                      disabled={!dataFetched || loading}
                      style={{
                        ...styles.measurementsButton,
                        ...(!dataFetched || loading
                          ? styles.measurementsButtonDisabled
                          : {}),
                      }}
                    >
                      Measurements
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>

          <div
            style={{
              ...styles.titleContainer,
              flexDirection: "column" as const,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <div
                style={{
                  ...styles.logo,
                  marginRight: "12px",
                  display: "flex",
                  alignItems: "center",
                }}
              >
                <img
                  src={lizard_logo}
                  alt="LizardMorph Logo"
                  style={{
                    height: "40px",
                    width: "40px",
                    objectFit: "contain",
                  }}
                />
              </div>
              <h2 style={styles.title}>
                {isFree
                  ? "Free Mode — Manual Landmarking"
                  : "Lizard Anolis X-Ray Auto-Annotator"}
              </h2>
            </div>
            <p style={styles.viewType}>
              View Type:{" "}
              {selectedViewType
                ? selectedViewType.charAt(0).toUpperCase() +
                  selectedViewType.slice(1)
                : ""}
            </p>
          </div>

          <div style={styles.rightSpacer}></div>
        </div>

        {dataError && (
          <span style={styles.errorMessage}>
            Error: {dataError.message}
          </span>
        )}
      </div>
    );
  }
}
