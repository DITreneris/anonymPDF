import React from 'react';
import {
  Paper,
  Typography,
  Box,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Button,
  Alert,
  AlertTitle,
  Skeleton,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Download as DownloadIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material';

interface RedactionCategory {
  [key: string]: number;
}

interface RedactionReport {
  totalRedactions: number;
  categories: RedactionCategory;
  confidence?: number;
  detectedLanguage?: string;
  rawReportString?: string;
  error?: string;
}

interface RedactionReportProps {
  report?: RedactionReport | string;
  downloadUrl?: string;
  onExportReport?: () => void;
}

const RedactionReport: React.FC<RedactionReportProps> = ({
  report,
  downloadUrl,
  onExportReport,
}) => {
  if (!report) {
    return (
      <Paper sx={{ p: 3 }}>
        <Typography variant="h5" gutterBottom>
          <Skeleton width="40%" />
        </Typography>
        <Skeleton variant="rectangular" height={60} sx={{ mb: 2 }} />
        <Skeleton variant="rectangular" height={40} sx={{ mb: 1 }} />
        <Skeleton variant="rectangular" height={40} />
        <Box sx={{ mt: 3, display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
            <Skeleton variant="rounded" width={150} height={36} />
            <Skeleton variant="rounded" width={220} height={36} />
        </Box>
      </Paper>
    );
  }

  if (typeof report === 'string') {
    return (
      <Paper sx={{ p: 3 }}>
        <Typography variant="h5" gutterBottom>
          Redaction Report
        </Typography>
        <Box sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', mt: 2 }}>
          {report}
        </Box>
      </Paper>
    );
  }

  const getConfidenceColor = (confidence: number): 'success' | 'warning' | 'error' => {
    if (confidence >= 0.9) return 'success';
    if (confidence >= 0.7) return 'warning';
    return 'error';
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Redaction Report
        </Typography>
        {typeof report === 'object' && report.totalRedactions !== undefined && (
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2, mb: 2 }}>
            <Box>
              <Typography variant="body1">
                Total Redactions: <strong>{report.totalRedactions}</strong>
              </Typography>
            </Box>
            {typeof report.confidence === 'number' && (
              <Box>
                <Typography variant="body1">
                  Confidence:{' '}
                  <Chip
                    label={`${(report.confidence * 100).toFixed(1)}%`}
                    color={getConfidenceColor(report.confidence)}
                    size="small"
                  />
                </Typography>
              </Box>
            )}
            {report.detectedLanguage && (
              <Box>
                <Typography variant="body1">
                  Detected Language: <strong>{report.detectedLanguage}</strong>
                </Typography>
              </Box>
            )}
          </Box>
        )}
      </Box>

      {typeof report === 'object' && report.categories && (
        <Accordion defaultExpanded>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">Categories</Typography>
          </AccordionSummary>
          <AccordionDetails>
            {Object.keys(report.categories).length > 0 ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {Object.entries(report.categories).map(([category, count]) => (
                  <Box key={category} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body2">{category}</Typography>
                    <Chip label={count} size="small" />
                  </Box>
                ))}
              </Box>
            ) : (
              <Typography variant="body2" color="text.secondary">No categories found or report is not structured.</Typography>
            )}
          </AccordionDetails>
        </Accordion>
      )}
      {typeof report === 'object' && report.rawReportString && (
          <Accordion sx={{mt: 2}}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">Raw Report Data</Typography>
            </AccordionSummary>
            <AccordionDetails>
                <Box sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', mt: 1, maxHeight: 200, overflowY: 'auto' }}>
                    {report.rawReportString}
                </Box>
            </AccordionDetails>
          </Accordion>
      )}
      {typeof report === 'object' && report.error && (
        <Alert severity="warning" sx={{ mt: 2}}>
          <AlertTitle>Report Issue</AlertTitle>
          {report.error}
        </Alert>
      )}

      <Box sx={{ mt: 3, display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
        {onExportReport && (
          <Button
            variant="outlined"
            startIcon={<AssessmentIcon />}
            onClick={onExportReport}
          >
            Export Report
          </Button>
        )}
        {downloadUrl && (
          <Button
            variant="contained"
            startIcon={<DownloadIcon />}
            href={downloadUrl}
            download
          >
            Download Anonymized PDF
          </Button>
        )}
      </Box>
    </Paper>
  );
};

export default RedactionReport; 