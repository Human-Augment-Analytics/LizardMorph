import React, { Component } from "react";
import { MeasurementsAndScalePanelStyles as styles } from "./MeasurementsAndScalePanel.style";
import type { Measurement } from "../models/Measurement";
import type { Point } from "../models/Point";
import type { ScaleSettings as ScaleSettingsType } from "../models/ScaleSettings";
import { UNITS } from "../models/ScaleSettings";

interface MeasurementsAndScalePanelProps {
  points: Point[];
  measurements: Measurement[];
  scaleSettings: ScaleSettingsType;
  onMeasurementsChange: (measurements: Measurement[]) => void;
  onScaleSettingsChange: (settings: ScaleSettingsType) => void;
  isModal?: boolean;
  onClose?: () => void;
}

export class MeasurementsAndScalePanel extends Component<MeasurementsAndScalePanelProps> {
  componentDidUpdate(prevProps: MeasurementsAndScalePanelProps): void {
    if (
      prevProps.scaleSettings !== this.props.scaleSettings ||
      prevProps.points !== this.props.points
    ) {
      const updatedMeasurements = this.props.measurements.map((measurement) => {
        const pointA = measurement.pointAId
          ? this.props.points.find((p) => p.id === measurement.pointAId) || null
          : null;
        const pointB = measurement.pointBId
          ? this.props.points.find((p) => p.id === measurement.pointBId) || null
          : null;

        return {
          ...measurement,
          calculatedDistance: this.calculateDistance(
            pointA,
            pointB,
            this.props.scaleSettings
          ),
        };
      });

      const hasChanges = updatedMeasurements.some(
        (m, i) =>
          m.calculatedDistance !== this.props.measurements[i]?.calculatedDistance
      );

      if (hasChanges) {
        this.props.onMeasurementsChange(updatedMeasurements);
      }
    }
  }

  private calculateDistance(
    pointA: Point | null,
    pointB: Point | null,
    scaleSettings: ScaleSettingsType
  ): number | null {
    if (!pointA || !pointB || !scaleSettings.pointAId || !scaleSettings.pointBId || scaleSettings.value === null || scaleSettings.value <= 0) {
      return null;
    }

    const pixelDistance = Math.sqrt(
      Math.pow(pointB.x - pointA.x, 2) + Math.pow(pointB.y - pointA.y, 2)
    );

    const scalePointA = this.props.points.find(
      (p) => p.id === scaleSettings.pointAId
    );
    const scalePointB = this.props.points.find(
      (p) => p.id === scaleSettings.pointBId
    );

    if (!scalePointA || !scalePointB) {
      return null;
    }

    const scalePixelDistance = Math.sqrt(
      Math.pow(scalePointB.x - scalePointA.x, 2) +
        Math.pow(scalePointB.y - scalePointA.y, 2)
    );

    if (scalePixelDistance === 0) {
      return null;
    }
    const realDistance = (pixelDistance / scalePixelDistance) * scaleSettings.value;

    return realDistance;
  }

  private handleAddMeasurement = (): void => {
    const newMeasurement: Measurement = {
      id: `measurement-${Date.now()}`,
      label: "",
      pointAId: null,
      pointBId: null,
      calculatedDistance: null,
    };

    this.props.onMeasurementsChange([
      ...this.props.measurements,
      newMeasurement,
    ]);
  };

  private handleDeleteMeasurement = (id: string): void => {
    this.props.onMeasurementsChange(
      this.props.measurements.filter((m) => m.id !== id)
    );
  };

  private handleMeasurementChange = (
    id: string,
    updates: Partial<Measurement>
  ): void => {
    const updatedMeasurements = this.props.measurements.map((m) => {
      if (m.id === id) {
        const updated = { ...m, ...updates };
        
        if (updates.pointAId !== undefined || updates.pointBId !== undefined) {
          const pointA = updated.pointAId
            ? this.props.points.find((p) => p.id === updated.pointAId) || null
            : null;
          const pointB = updated.pointBId
            ? this.props.points.find((p) => p.id === updated.pointBId) || null
            : null;

          updated.calculatedDistance = this.calculateDistance(
            pointA,
            pointB,
            this.props.scaleSettings
          );
        }

        return updated;
      }
      return m;
    });

    this.props.onMeasurementsChange(updatedMeasurements);
  };

  render() {
    const { points, measurements, scaleSettings, onScaleSettingsChange, isModal, onClose } = this.props;

    const handlePointAChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
      const pointAId = e.target.value ? parseInt(e.target.value, 10) : null;
      onScaleSettingsChange({
        ...scaleSettings,
        pointAId,
      });
    };

    const handlePointBChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
      const pointBId = e.target.value ? parseInt(e.target.value, 10) : null;
      onScaleSettingsChange({
        ...scaleSettings,
        pointBId,
      });
    };

    const handleValueChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value ? parseFloat(e.target.value) : null;
      onScaleSettingsChange({
        ...scaleSettings,
        value,
      });
    };

    const handleUnitsChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
      onScaleSettingsChange({
        ...scaleSettings,
        units: e.target.value,
      });
    };

    const scaleSettingsContent = (
      <>
        <div style={styles.formGroup}>
          <label style={styles.label}>Point A:</label>
          <select
            style={styles.select}
            value={scaleSettings.pointAId ?? ""}
            onChange={handlePointAChange}
          >
            <option value="">Select landmark...</option>
            {points.map((point) => (
              <option key={point.id} value={point.id}>
                Landmark {point.id}
              </option>
            ))}
          </select>
        </div>

        <div style={styles.formGroup}>
          <label style={styles.label}>Point B:</label>
          <select
            style={styles.select}
            value={scaleSettings.pointBId ?? ""}
            onChange={handlePointBChange}
          >
            <option value="">Select landmark...</option>
            {points.map((point) => (
              <option key={point.id} value={point.id}>
                Landmark {point.id}
              </option>
            ))}
          </select>
        </div>

        <div style={styles.formGroup}>
          <label style={styles.label}>Value:</label>
          <input
            type="number"
            style={styles.input}
            value={scaleSettings.value ?? ""}
            onChange={handleValueChange}
            placeholder="Enter known distance"
            step="any"
            min="0"
          />
        </div>

        <div style={styles.formGroup}>
          <label style={styles.label}>Units:</label>
          <select
            style={styles.select}
            value={scaleSettings.units}
            onChange={handleUnitsChange}
          >
            {UNITS.map((unit) => (
              <option key={unit.value} value={unit.value}>
                {unit.label}
              </option>
            ))}
          </select>
        </div>

        {scaleSettings.pointAId &&
          scaleSettings.pointBId &&
          scaleSettings.value !== null &&
          scaleSettings.value > 0 && (
            <div style={styles.infoText}>
              Scale set: {scaleSettings.value} {scaleSettings.units} between
              Landmark {scaleSettings.pointAId} and Landmark{" "}
              {scaleSettings.pointBId}
            </div>
          )}
      </>
    );

    const measurementsContent = (
      <>
        {measurements.length === 0 ? (
          <div style={styles.emptyState}>
            No measurements yet. Click "Add Measurement" to create one.
          </div>
        ) : (
          measurements.map((measurement, index) => {
            const pointA = measurement.pointAId
              ? points.find((p) => p.id === measurement.pointAId)
              : null;
            const pointB = measurement.pointBId
              ? points.find((p) => p.id === measurement.pointBId)
              : null;

            let calculatedDistance = measurement.calculatedDistance;
            if (pointA && pointB) {
              calculatedDistance = this.calculateDistance(
                pointA,
                pointB,
                scaleSettings
              );
            }

            return (
              <div key={measurement.id} style={styles.measurementItem}>
                <div style={styles.measurementHeader}>
                  <span style={styles.measurementLabel}>
                    Measurement #{index + 1}
                  </span>
                  <button
                    style={styles.deleteButton}
                    onClick={() => this.handleDeleteMeasurement(measurement.id)}
                  >
                    Delete
                  </button>
                </div>

                <div style={styles.formGroup}>
                  <label style={styles.label}>Label:</label>
                  <input
                    type="text"
                    style={styles.input}
                    value={measurement.label}
                    onChange={(e) =>
                      this.handleMeasurementChange(measurement.id, {
                        label: e.target.value,
                      })
                    }
                    placeholder="e.g., Head width"
                  />
                </div>

                <div style={styles.formGroup}>
                  <label style={styles.label}>Point A:</label>
                  <select
                    style={styles.select}
                    value={measurement.pointAId ?? ""}
                    onChange={(e) =>
                      this.handleMeasurementChange(measurement.id, {
                        pointAId: e.target.value
                          ? parseInt(e.target.value, 10)
                          : null,
                      })
                    }
                  >
                    <option value="">Select landmark...</option>
                    {points.map((point) => (
                      <option key={point.id} value={point.id}>
                        Landmark {point.id}
                      </option>
                    ))}
                  </select>
                </div>

                <div style={styles.formGroup}>
                  <label style={styles.label}>Point B:</label>
                  <select
                    style={styles.select}
                    value={measurement.pointBId ?? ""}
                    onChange={(e) =>
                      this.handleMeasurementChange(measurement.id, {
                        pointBId: e.target.value
                          ? parseInt(e.target.value, 10)
                          : null,
                      })
                    }
                  >
                    <option value="">Select landmark...</option>
                    {points.map((point) => (
                      <option key={point.id} value={point.id}>
                        Landmark {point.id}
                      </option>
                    ))}
                  </select>
                </div>

                {calculatedDistance !== null && (
                  <div style={styles.distanceDisplay}>
                    Distance: {calculatedDistance.toFixed(3)} {scaleSettings.units}
                    {measurement.label && ` (${measurement.label})`}
                  </div>
                )}
              </div>
            );
          })
        )}

        <button
          style={styles.addButton}
          onClick={this.handleAddMeasurement}
        >
          + Add Measurement
        </button>
      </>
    );

    const modalContent = (
      <>
        <div style={styles.sectionTitle}>Set Scale</div>
        {scaleSettingsContent}
        <div style={styles.sectionTitle}>Measurements</div>
        {measurementsContent}
      </>
    );

    if (isModal) {
      return (
        <>
          <div
            style={styles.modalOverlay}
            onClick={onClose}
          />
          <div
            style={styles.modalContent}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={styles.modalHeader}>
              <h3 style={{ margin: 0 }}>Scale and Measurements</h3>
              <button
                style={styles.closeButton}
                onClick={onClose}
                aria-label="Close"
              >
                ×
              </button>
            </div>
            <div style={styles.modalBody}>
              {modalContent}
            </div>
          </div>
        </>
      );
    }

    return (
      <div style={styles.container}>
        {modalContent}
      </div>
    );
  }
}
