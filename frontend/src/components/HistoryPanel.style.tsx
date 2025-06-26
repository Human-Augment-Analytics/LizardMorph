import type { CSSProperties } from "react";

export class HistoryPanelStyles {
  static readonly historyContainer: CSSProperties = {
    width: "27vw",
    borderRight: "1px solid #ccc",
    overflowY: "auto",
    padding: "10px",
  };

  static readonly historyTableContainer: CSSProperties = {
    maxHeight: "calc(100vh - 350px)",
    overflowY: "auto",
    border: "1px solid #ddd",
    borderRadius: "4px",
  };

  static readonly historyTable: CSSProperties = {
    width: "100%",
    borderCollapse: "collapse",
  };

  static readonly historyTableHeader: CSSProperties = {
    backgroundColor: "#f3f3f3",
  };

  static readonly historyTableHeaderCell: CSSProperties = {
    padding: "8px",
    textAlign: "left",
    borderBottom: "1px solid #ddd",
    color: "black",
  };

  static readonly historyTableRow: CSSProperties = {
    cursor: "pointer",
  };

  static readonly historyTableRowSelected: CSSProperties = {
    backgroundColor: "#f0f0f0",
  };

  static readonly historyTableCell: CSSProperties = {
    padding: "8px",
    borderBottom: "1px solid #eee",
    whiteSpace: "nowrap",
    overflow: "hidden",
    textOverflow: "ellipsis",
    color: "white",
  };

  static readonly historyTableCellSelected: CSSProperties = {
    fontWeight: "bold",
    color: "black",
  };

  static readonly historyTableEmptyCell: CSSProperties = {
    padding: "10px",
    textAlign: "center",
    color: "#666",
  };
}
