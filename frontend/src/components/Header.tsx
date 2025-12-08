import React, { Component } from "react";
import { HeaderStyles } from "./Header.style";

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
    } = this.props;

    const { isMenuOpen } = this.state;

    return (
      <div style={HeaderStyles.header}>
        <div style={HeaderStyles.infoBox}>
          <div style={HeaderStyles.infoBoxContent}>
            <p style={HeaderStyles.infoBoxParagraph}>
              Made with ❤️ by the Human Augmented Analytics Group (HAAG)
            </p>
            <p style={HeaderStyles.infoBoxParagraph}>
              In Partnership with Dr. Stroud
            </p>
            <p style={HeaderStyles.infoBoxItalic}>
              Georgia Institute of Technology
            </p>
            <a
              href="https://github.com/Human-Augment-Analytics/Lizard-CV-Web-App"
              target="_blank"
              rel="noopener noreferrer"
              style={HeaderStyles.infoBoxLink}
            >
              View on GitHub
            </a>
            <div style={HeaderStyles.lizardCount}>
              <strong>Number of Lizards Analyzed: {lizardCount}</strong>
            </div>
          </div>
        </div>
        <div style={HeaderStyles.mainContent}>
          <div style={HeaderStyles.buttonContainer}>
            <label
              htmlFor="file-upload"
              style={{
                ...HeaderStyles.uploadButton,
                ...(loading ? HeaderStyles.uploadButtonDisabled : {}),
              }}
            >
              {loading ? "Uploading..." : "Upload X-Ray Images"}
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
                ...HeaderStyles.exportButton,
                ...(!dataFetched || loading
                  ? HeaderStyles.exportButtonDisabled
                  : {}),
              }}
            >
              Export All Data
            </button>

            <div style={HeaderStyles.dropdownContainer}>
              <button
                onClick={this.toggleMenu}
                style={HeaderStyles.dropdownButton}
              >
                More...
              </button>
              {isMenuOpen && (
                <div style={HeaderStyles.dropdownContent}>
                  <button
                    onClick={onClearHistory}
                    disabled={loading}
                    style={{
                      ...HeaderStyles.clearHistoryButton,
                      ...(loading
                        ? HeaderStyles.clearHistoryButtonDisabled
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
                        ...HeaderStyles.measurementsButton,
                        ...(!dataFetched || loading
                          ? HeaderStyles.measurementsButtonDisabled
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
              ...HeaderStyles.titleContainer,
              cursor: "pointer",
              flexDirection: "column" as const,
            }}
            onClick={this.props.onBackToSelection}
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
                  ...HeaderStyles.logo,
                  fontSize: "40px",
                  marginRight: "12px",
                }}
              >
                {this.props.selectedViewType === "lateral" ? "🦖" : this.props.selectedViewType === "toepads" ? "🦶" : "🦎"}
              </div>
              <h2 style={HeaderStyles.title}>
                Lizard Anolis X-Ray Auto-Annotator
              </h2>
            </div>
            <p style={HeaderStyles.viewType}>
              View Type:{" "}
              {this.props.selectedViewType
                ? this.props.selectedViewType.charAt(0).toUpperCase() +
                  this.props.selectedViewType.slice(1)
                : ""}
            </p>
          </div>

          <div style={HeaderStyles.rightSpacer}></div>
        </div>

        {dataError && (
          <span style={HeaderStyles.errorMessage}>
            Error: {dataError.message}
          </span>
        )}
      </div>
    );
  }
}
