import React, { Component } from "react";
import { HeaderStyles } from "./Header.style";

interface HeaderProps {
  lizardCount: number;
  loading: boolean;
  dataFetched: boolean;
  dataError: Error | null;
  onUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onExportAll: () => void;
  onClearHistory: () => void;
}

export class Header extends Component<HeaderProps> {
  render() {
    const {
      lizardCount,
      loading,
      dataFetched,
      dataError,
      onUpload,
      onExportAll,
      onClearHistory,
    } = this.props;

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
            <p style={HeaderStyles.infoBoxParagraph}>
              Author: Mercedes Quintana
            </p>
            <p style={HeaderStyles.infoBoxParagraph}>
              AI Engineer: Anthony Trevino
            </p>
            <p style={HeaderStyles.infoBoxItalic}>
              Georgia Institute of Technology - Spring 2025
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

            <button
              onClick={onClearHistory}
              disabled={loading}
              style={{
                ...HeaderStyles.clearHistoryButton,
                ...(loading ? HeaderStyles.clearHistoryButtonDisabled : {}),
              }}
            >
              Clear History
            </button>
          </div>

          <div style={HeaderStyles.titleContainer}>
            <img
              src="/android-chrome-192x192.png"
              alt="Lizard Logo"
              style={HeaderStyles.logo}
            />
            <h2 style={HeaderStyles.title}>
              Lizard Anolis X-Ray Auto-Annotator
            </h2>
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
