import React from "react";
import { PointsPanelStyles } from "./PointsPanel.style";

interface Point {
  id: number;
  x: number;
  y: number;
}

interface PointsPanelProps {
  dataFetched: boolean;
  selectedPoint: Point | null;
  scatterData: Point[];
  imageFilename: string;
  currentImageIndex: number;
  totalImages: number;
  loading: boolean;
  onPointSelect: (point: Point) => void;
  onSaveAnnotations: () => void;
  formatCoord: (coord: number) => string;
}

class PointsPanel extends React.Component<PointsPanelProps> {
  render() {
    const {
      dataFetched,
      selectedPoint,
      scatterData,
      imageFilename,
      currentImageIndex,
      totalImages,
      loading,
      onPointSelect,
      onSaveAnnotations,
      formatCoord,
    } = this.props;

    if (!dataFetched) {
      return null;
    }
    return (
      <div style={PointsPanelStyles.pointsContainer}>
        <div style={PointsPanelStyles.pointsHeader}>
          <h3 style={PointsPanelStyles.selectedPointHeader}>Landmark Points</h3>
          <button
            onClick={onSaveAnnotations}
            disabled={loading}
            style={{
              ...PointsPanelStyles.saveButton,
              ...(loading ? PointsPanelStyles.saveButtonDisabled : {}),
            }}
          >
            {loading ? "Saving..." : "Save Annotations"}
          </button>
        </div>

        {selectedPoint && (
          <div style={PointsPanelStyles.selectedPointDetails}>
            <h4 style={PointsPanelStyles.selectedPointHeader}>
              Selected Point Details
            </h4>
            <p>
              <strong>Point {selectedPoint.id}</strong>
            </p>
            <p>
              <strong>X coordinate:</strong> {formatCoord(selectedPoint.x)}
            </p>
            <p>
              <strong>Y coordinate:</strong> {formatCoord(selectedPoint.y)}
            </p>
            <div style={PointsPanelStyles.selectedPointInfo}>
              <p>Image: {imageFilename}</p>
              <p>
                Image {currentImageIndex + 1} of {totalImages}
              </p>
            </div>
          </div>
        )}

        <p>
          Click on a row to select a point. Selected point is highlighted in
          yellow.
        </p>
        <table style={PointsPanelStyles.pointsTable}>
          <thead>
            <tr style={PointsPanelStyles.pointsTableHeader}>
              <th style={PointsPanelStyles.pointsTableHeaderCell}>Point ID</th>
              <th style={PointsPanelStyles.pointsTableHeaderCell}>X</th>
              <th style={PointsPanelStyles.pointsTableHeaderCell}>Y</th>
            </tr>
          </thead>
          <tbody>
            {scatterData.map((point) => (
              <tr
                key={point.id}
                onClick={() => onPointSelect(point)}
                style={{
                  ...PointsPanelStyles.pointsTableRow,
                  ...(selectedPoint && selectedPoint.id === point.id
                    ? PointsPanelStyles.pointsTableRowSelected
                    : {}),
                }}
              >
                <td style={PointsPanelStyles.pointsTableCell}>
                  Point {point.id}
                </td>
                <td style={PointsPanelStyles.pointsTableCell}>
                  {formatCoord(point.x)}
                </td>
                <td style={PointsPanelStyles.pointsTableCell}>
                  {formatCoord(point.y)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
}

export { PointsPanel };
export default PointsPanel;
