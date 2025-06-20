import React, { useState } from 'react';
import apiClient from '../utils/api';
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
  ThumbUp,
  ThumbDown
} from '@mui/icons-material';

interface RedactionDetail {
  text: string;
  category: string;
  confidence: number;
  start: number;
  end: number;
}

interface RedactionCategory {
  [key: string]: number;
}

export interface RedactionReport {
  totalRedactions: number;
  categories: RedactionCategory;
  details?: RedactionDetail[];
  confidence?: number;
  detectedLanguage?: string;
  rawReportString?: string;
  error?: string;
}

interface RedactionReportProps {
  documentId?: number;
  report?: RedactionReport | null;
  downloadUrl?: string;
}

const RedactionReport: React.FC<RedactionReportProps> = ({
  documentId,
  report,
  downloadUrl,
}) => {
  const [feedbackSent, setFeedbackSent] = useState<Set<number>>(new Set());

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

  const getConfidenceColor = (confidence: number): 'success' | 'warning' | 'error' => {
    if (confidence >= 0.9) return 'success';
    if (confidence >= 0.7) return 'warning';
    return 'error';
  };

  const exportReportAsJSON = () => {
    if (!report || typeof report === 'string') return;

    // Create a blob from the report data
    const jsonString = JSON.stringify(report, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    // Create a temporary link to trigger the download
    const link = document.createElement('a');
    link.href = url;
    link.download = `redaction-report-${documentId || 'details'}.json`;
    document.body.appendChild(link);
    link.click();

    // Clean up
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleFeedback = async (item: RedactionDetail, isCorrect: boolean, index: number) => {
    if (!documentId) {
      console.error("Cannot send feedback: documentId is missing.");
      return;
    }

    const payload = {
      document_id: documentId,
      feedback_items: [
        {
          text_segment: item.text,
          original_category: item.category,
          is_correct: isCorrect,
        },
      ],
    };

    try {
      await apiClient.post('/feedback', payload);
      // Visually confirm feedback was sent by adding the item's index to a set
      setFeedbackSent(prev => new Set(prev).add(index));
    } catch (error) {
      console.error("Failed to send feedback:", error);
      // Optionally, show an error message to the user
    }
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

      {/* Details Accordion */}
      {typeof report === 'object' && report.details && report.details.length > 0 && (
        <Accordion sx={{ mt: 2 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="h6">Redaction Details</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {report.details.map((item, index) => (
                <Paper 
                  key={index} 
                  variant="outlined" 
                  sx={{ 
                    p: 2, 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    opacity: feedbackSent.has(index) ? 0.5 : 1, // Visual feedback
                    transition: 'opacity 0.3s ease-in-out',
                  }}
                >
                  <Box>
                    <Typography variant="body1"><strong>Text:</strong> "{item.text}"</Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Category:</strong> {item.category} | <strong>Confidence:</strong>{' '}
                      <Chip
                        label={`${(item.confidence * 100).toFixed(1)}%`}
                        color={getConfidenceColor(item.confidence)}
                        size="small"
                      />
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                      variant="outlined"
                      size="small"
                      startIcon={<ThumbUp />}
                      color="success"
                      onClick={() => handleFeedback(item, true, index)}
                      disabled={feedbackSent.has(index)}
                    >
                      Correct
                    </Button>
                    <Button
                      variant="outlined"
                      size="small"
                      startIcon={<ThumbDown />}
                      color="error"
                      onClick={() => handleFeedback(item, false, index)}
                      disabled={feedbackSent.has(index)}
                    >
                      Incorrect
                    </Button>
                  </Box>
                </Paper>
              ))}
            </Box>
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
        {report && typeof report !== 'string' && (
          <Button
            variant="outlined"
            startIcon={<AssessmentIcon />}
            onClick={exportReportAsJSON}
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