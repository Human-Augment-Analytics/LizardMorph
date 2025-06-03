import type { CSSProperties } from "react";

export class PointsPanelStyles {
  static readonly pointsContainer: CSSProperties = {
    flex: 1,
    borderLeft: "1px solid #ccc",
    padding: "10px",
    overflowY: "auto",
  };

  static readonly pointsHeader: CSSProperties = {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "10px",
  };

  static readonly saveButton: CSSProperties = {
    padding: "8px 15px",
    backgroundColor: "#ff9800",
    color: "white",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
    fontWeight: "bold",
    fontSize: "14px",
    boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
  };

  static readonly saveButtonDisabled: CSSProperties = {
    backgroundColor: "#ccc",
    cursor: "not-allowed",
  };

  static readonly selectedPointDetails: CSSProperties = {
    marginBottom: "20px",
    padding: "10px",
    backgroundColor: "#f9f9f9",
    borderRadius: "4px",
    border: "1px solid #ddd",
  };

  static readonly selectedPointHeader: CSSProperties = {
    marginTop: 0,
  };

  static readonly selectedPointInfo: CSSProperties = {
    marginTop: "10px",
    fontSize: "0.9em",
    color: "#666",
  };

  static readonly pointsTable: CSSProperties = {
    width: "100%",
    borderCollapse: "collapse",
  };

  static readonly pointsTableHeader: CSSProperties = {
    backgroundColor: "#f3f3f3",
  };

  static readonly pointsTableHeaderCell: CSSProperties = {
    padding: "8px",
    textAlign: "center",
    borderBottom: "1px solid #ddd",
    color: "black",
  };

  static readonly pointsTableRow: CSSProperties = {
    cursor: "pointer",
    transition: "background-color 0.2s",
  };

  static readonly pointsTableRowSelected: CSSProperties = {
    backgroundColor: "#ffff99",
    color: "black",
  };

  static readonly pointsTableCell: CSSProperties = {
    padding: "8px",
    borderBottom: "1px solid #ddd",
  };
}
