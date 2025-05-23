import type { CSSProperties } from 'react';

export class MainViewStyles {
  static readonly container: CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    minWidth: '1400px',
  };

  static readonly header: CSSProperties = {
    padding: '15px',
    borderBottom: '1px solid #ccc',
    textAlign: 'center',
    minHeight: '220px',
    position: 'relative',
  };

  static readonly infoBox: CSSProperties = {
    position: 'absolute',
    top: '10px',
    right: '15px',
    padding: '12px',
    backgroundColor: '#f8f9fa',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
    border: '1px solid #dee2e6',
    textAlign: 'right',
    minWidth: '300px',
  };

  static readonly infoBoxContent: CSSProperties = {
    fontSize: '0.9em',
    color: '#495057',
  };

  static readonly infoBoxParagraph: CSSProperties = {
    margin: '4px 0',
  };

  static readonly infoBoxItalic: CSSProperties = {
    margin: '4px 0',
    fontStyle: 'italic',
  };

  static readonly infoBoxLink: CSSProperties = {
    color: '#0056b3',
    textDecoration: 'none',
  };

  static readonly lizardCount: CSSProperties = {
    marginTop: '5px',
    color: '#0056b3',
  };

  static readonly mainContent: CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    maxWidth: '1200px',
    margin: '15px auto 0',
    padding: '0 20px',
  };

  static readonly buttonContainer: CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-start',
    gap: '15px',
    width: '220px',
    marginTop: '20px',
  };

  static readonly uploadButton: CSSProperties = {
    padding: '12px 20px',
    backgroundColor: '#2196F3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    display: 'inline-block',
    fontWeight: 'bold',
    fontSize: '14px',
    textAlign: 'center',
    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
    width: '100%',
    boxSizing: 'border-box',
  };

  static readonly uploadButtonDisabled: CSSProperties = {
    cursor: 'not-allowed',
    opacity: 0.7,
  };

  static readonly exportButton: CSSProperties = {
    padding: '12px 20px',
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    width: '100%',
    boxSizing: 'border-box',
    fontWeight: 'bold',
    fontSize: '14px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
  };

  static readonly exportButtonDisabled: CSSProperties = {
    cursor: 'not-allowed',
    opacity: 0.7,
  };

  static readonly titleContainer: CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };

  static readonly logo: CSSProperties = {
    height: '40px',
    marginRight: '12px',
  };

  static readonly title: CSSProperties = {
    margin: 0,
    whiteSpace: 'nowrap',
  };

  static readonly rightSpacer: CSSProperties = {
    width: '220px',
  };

  static readonly errorMessage: CSSProperties = {
    color: 'red',
  };

  static readonly navigationControls: CSSProperties = {
    marginTop: '10px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '10px',
  };

  static readonly navButton: CSSProperties = {
    padding: '8px 12px',
    backgroundColor: '#f0f0f0',
    color: 'black',
    border: 'none',
    borderRadius: '4px',
    fontSize: '14px',
    cursor: 'pointer',
  };

  static readonly navButtonDisabled: CSSProperties = {
    cursor: 'not-allowed',
  };

  static readonly imageVersionButtons: CSSProperties = {
    marginTop: '10px',
    display: 'flex',
    gap: '10px',
    justifyContent: 'center',
  };

  static readonly versionButton: CSSProperties = {
    padding: '8px 12px',
    backgroundColor: '#f0f0f0',
    color: 'black',
    border: 'none',
    borderRadius: '4px',
    fontSize: '14px',
    cursor: 'pointer',
    opacity: 1,
  };

  static readonly versionButtonActive: CSSProperties = {
    backgroundColor: '#2196F3',
    color: 'white',
  };

  static readonly versionButtonDisabled: CSSProperties = {
    cursor: 'not-allowed',
    opacity: 0.7,
  };

  static readonly mainContentArea: CSSProperties = {
    display: 'flex',
    flex: 1,
    overflow: 'hidden',
  };

  static readonly historyContainer: CSSProperties = {
    width: '27vw',
    borderRight: '1px solid #ccc',
    overflowY: 'auto',
    padding: '10px',
  };

  static readonly historyTableContainer: CSSProperties = {
    maxHeight: 'calc(100vh - 250px)',
    overflowY: 'auto',
    border: '1px solid #ddd',
    borderRadius: '4px',
  };

  static readonly historyTable: CSSProperties = {
    width: '100%',
    borderCollapse: 'collapse',
  };

  static readonly historyTableHeader: CSSProperties = {
    backgroundColor: '#f3f3f3',
  };

  static readonly historyTableHeaderCell: CSSProperties = {
    padding: '8px',
    textAlign: 'left',
    borderBottom: '1px solid #ddd',
    color: 'black',
  };

  static readonly historyTableRow: CSSProperties = {
    cursor: 'pointer',
  };

  static readonly historyTableRowSelected: CSSProperties = {
    backgroundColor: '#f0f0f0',
  };

  static readonly historyTableCell: CSSProperties = {
    padding: '8px',
    borderBottom: '1px solid #eee',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    color: 'black',
  };

  static readonly historyTableCellSelected: CSSProperties = {
    fontWeight: 'bold',
  };

  static readonly historyTableEmptyCell: CSSProperties = {
    padding: '10px',
    textAlign: 'center',
    color: '#666',
  };

  static readonly svgContainer: CSSProperties = {
    flex: 3,
    overflow: 'auto',
    position: 'relative',
  };

  static readonly placeholderMessage: CSSProperties = {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    textAlign: 'center',
    color: '#666',
  };

  static readonly placeholderSubtext: CSSProperties = {
    fontSize: '0.9em',
  };

  static readonly svg: CSSProperties = {
    display: 'block',
    margin: '0 auto',
    boxShadow: '0 0 5px rgba(0,0,0,0.2)',
  };

  static readonly svgWithData: CSSProperties = {
    backgroundColor: '#f9f9f9',
  };

  static readonly loadingOverlay: CSSProperties = {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    backgroundColor: 'rgba(255,255,255,0.8)',
    padding: '15px',
    borderRadius: '5px',
    boxShadow: '0 0 10px rgba(0,0,0,0.1)',
  };

  static readonly errorOverlay: CSSProperties = {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    backgroundColor: 'rgba(255,220,220,0.9)',
    padding: '15px',
    borderRadius: '5px',
    color: 'red',
    boxShadow: '0 0 10px rgba(255,0,0,0.2)',
  };

  static readonly pointsContainer: CSSProperties = {
    flex: 1,
    borderLeft: '1px solid #ccc',
    padding: '10px',
    overflowY: 'auto',
  };

  static readonly pointsHeader: CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '10px',
  };

  static readonly saveButton: CSSProperties = {
    padding: '8px 15px',
    backgroundColor: '#ff9800',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontWeight: 'bold',
    fontSize: '14px',
    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  };

  static readonly saveButtonDisabled: CSSProperties = {
    backgroundColor: '#ccc',
    cursor: 'not-allowed',
  };

  static readonly selectedPointDetails: CSSProperties = {
    marginBottom: '20px',
    padding: '10px',
    backgroundColor: '#f9f9f9',
    borderRadius: '4px',
    border: '1px solid #ddd',
  };

  static readonly selectedPointHeader: CSSProperties = {
    marginTop: 0,
  };

  static readonly selectedPointInfo: CSSProperties = {
    marginTop: '10px',
    fontSize: '0.9em',
    color: '#666',
  };

  static readonly pointsTable: CSSProperties = {
    width: '100%',
    borderCollapse: 'collapse',
  };

  static readonly pointsTableHeader: CSSProperties = {
    backgroundColor: '#f3f3f3',
  };

  static readonly pointsTableHeaderCell: CSSProperties = {
    padding: '8px',
    textAlign: 'center',
    borderBottom: '1px solid #ddd',
    color: 'black',
  };

  static readonly pointsTableRow: CSSProperties = {
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  };

  static readonly pointsTableRowSelected: CSSProperties = {
    backgroundColor: '#ffff99',
    color: 'black',
  };

  static readonly pointsTableCell: CSSProperties = {
    padding: '8px',
    borderBottom: '1px solid #ddd',
  };
}