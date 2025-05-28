import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Button,
  Alert,
  AlertTitle,
  Stack,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Download as DownloadIcon,
  Assessment as AssessmentIcon,
  Security,
  Speed,
  Info,
  BarChart as BarChartIcon,
} from '@mui/icons-material';

interface RedactionCategory {
  [key: string]: number;
}

interface RedactionReportData {
  title?: string;
  detectedLanguage?: string;
  totalRedactions: number;
  categories: RedactionCategory;
  confidence?: number;
  error?: string;
}

interface StatisticsPanelProps {
  redactionReport?: RedactionReportData | null;
  downloadUrl?: string;
  onExportReport?: () => void;
  theme: any; // Material-UI theme
}

const StatisticsPanel: React.FC<StatisticsPanelProps> = ({
  redactionReport,
  downloadUrl,
  onExportReport,
  theme,
}) => {
  // Define accordion sections for "How it works"
  const accordionSections = [
    {
      id: 'pii',
      title: 'Personal Information Detection',
      icon: <Info sx={{ fontSize: 20, color: 'white' }} />,
      iconColor: theme.palette.info.main,
      description: 'We automatically detect and redact Personally Identifiable Information (PII) - any data that could identify a specific individual:',
      items: [
        'Names and addresses',
        'Phone numbers and emails',
        'ID numbers and VAT codes',
        'Bank account details',
        'Social security numbers',
        'Passport and license numbers',
      ],
    },
    {
      id: 'security',
      title: 'Security & Privacy',
      icon: <Security sx={{ fontSize: 20, color: 'white' }} />,
      iconColor: theme.palette.success.main,
      items: [
        'All processing happens locally on your computer',
        'No data is sent to external servers',
        'Original files are never modified',
        'You maintain full control of your documents',
        'No internet connection required for processing',
        'Complete data sovereignty and privacy',
      ],
    },
    {
      id: 'processing',
      title: 'Fast Processing',
      icon: <Speed sx={{ fontSize: 20, color: 'white' }} />,
      iconColor: theme.palette.warning.main,
      description: 'Most documents are processed in under 10 seconds. Processing time depends on file size and complexity.',
      items: [
        'Optimized for documents up to 10MB',
        'Real-time progress tracking',
        'Automatic language detection',
        'Comprehensive redaction reporting',
        'Multiple file format support',
      ],
    },
  ];

  const getConfidenceColor = (confidence: number): 'success' | 'warning' | 'error' => {
    if (confidence >= 0.9) return 'success';
    if (confidence >= 0.7) return 'warning';
    return 'error';
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Redaction Statistics Section - Only show when we have results */}
      {redactionReport && (
        <Paper 
          sx={{ 
            p: 3, 
            mb: 3,
            border: '1px solid #E5E7EB',
            borderRadius: '12px',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
            <Box
              sx={{
                width: 48,
                height: 48,
                borderRadius: '12px',
                backgroundColor: theme.palette.success.main,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: `0 4px 6px ${theme.palette.success.main}33`,
              }}
            >
              <BarChartIcon sx={{ fontSize: 24, color: 'white' }} />
            </Box>
            <Box>
              <Typography variant="h6" color="text.primary" sx={{ fontWeight: 600 }}>
                Redaction Statistics
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Summary of what was detected and redacted
              </Typography>
            </Box>
          </Box>

          {/* Key Statistics */}
          <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
            <Box sx={{ flex: 1 }}>
              <Box sx={{ textAlign: 'center', p: 2, backgroundColor: '#F9FAFB', borderRadius: '8px' }}>
                <Typography variant="h4" color="primary.main" sx={{ fontWeight: 700 }}>
                  {redactionReport.totalRedactions}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total Redactions
                </Typography>
              </Box>
            </Box>
            <Box sx={{ flex: 1 }}>
              <Box sx={{ textAlign: 'center', p: 2, backgroundColor: '#F9FAFB', borderRadius: '8px' }}>
                <Typography variant="h4" color="primary.main" sx={{ fontWeight: 700 }}>
                  {Object.keys(redactionReport.categories).length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  PII Categories
                </Typography>
              </Box>
            </Box>
          </Box>

          {/* Additional Info */}
          {(redactionReport.confidence !== undefined || redactionReport.detectedLanguage) && (
            <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
              {redactionReport.confidence !== undefined && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Confidence:
                  </Typography>
                  <Chip
                    label={`${(redactionReport.confidence * 100).toFixed(1)}%`}
                    color={getConfidenceColor(redactionReport.confidence)}
                    size="small"
                  />
                </Box>
              )}
              {redactionReport.detectedLanguage && (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    Language:
                  </Typography>
                  <Chip
                    label={redactionReport.detectedLanguage}
                    variant="outlined"
                    size="small"
                  />
                </Box>
              )}
            </Box>
          )}

          {/* Categories Breakdown */}
          {Object.keys(redactionReport.categories).length > 0 && (
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                  Detected PII Categories
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Stack spacing={1}>
                  {Object.entries(redactionReport.categories)
                    .sort(([,a], [,b]) => b - a) // Sort by count descending
                    .map(([category, count]) => (
                    <Box 
                      key={category} 
                      sx={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        alignItems: 'center',
                        p: 1,
                        backgroundColor: '#F9FAFB',
                        borderRadius: '6px',
                      }}
                    >
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {category}
                      </Typography>
                      <Chip 
                        label={count} 
                        size="small" 
                        color="primary"
                        variant="outlined"
                      />
                    </Box>
                  ))}
                </Stack>
              </AccordionDetails>
            </Accordion>
          )}

          {/* Error Display */}
          {redactionReport.error && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              <AlertTitle>Report Issue</AlertTitle>
              {redactionReport.error}
            </Alert>
          )}

          {/* Action Buttons */}
          <Box sx={{ mt: 3, display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
            {onExportReport && (
              <Button
                variant="outlined"
                startIcon={<AssessmentIcon />}
                onClick={onExportReport}
                sx={{ borderRadius: '8px', textTransform: 'none' }}
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
                sx={{ borderRadius: '8px', textTransform: 'none' }}
              >
                Download PDF
              </Button>
            )}
          </Box>
        </Paper>
      )}

      {/* How It Works Section - Always visible */}
      <Box sx={{ flex: 1 }}>
        <Box sx={{ mb: 3 }}>
          <Typography variant="h6" color="primary.main" sx={{ fontWeight: 600, mb: 1 }}>
            How It Works
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Learn about PII detection, security features, and processing capabilities
          </Typography>
        </Box>
        
        <Stack spacing={2}>
          {accordionSections.map((section) => (
            <Accordion 
              key={section.id}
              defaultExpanded={section.id === 'pii'} // Expand PII section by default
              sx={{
                border: '1px solid #E5E7EB',
                borderRadius: '12px !important',
                boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
                '&:before': { display: 'none' },
              }}
            >
              <AccordionSummary 
                expandIcon={<ExpandMoreIcon />}
                sx={{ 
                  borderRadius: '12px',
                  '&.Mui-expanded': {
                    borderBottomLeftRadius: 0,
                    borderBottomRightRadius: 0,
                  }
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Box
                    sx={{
                      width: 40,
                      height: 40,
                      borderRadius: '10px',
                      backgroundColor: section.iconColor,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: `0 4px 6px ${section.iconColor}33`,
                    }}
                  >
                    {section.icon}
                  </Box>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    {section.title}
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails sx={{ pt: 0 }}>
                {section.description && (
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {section.description}
                  </Typography>
                )}
                <Stack spacing={1}>
                  {section.items.map((item, index) => (
                    <Box key={index} sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                      <Box
                        sx={{
                          width: 6,
                          height: 6,
                          borderRadius: '50%',
                          backgroundColor: 'primary.main',
                          mt: 0.75,
                          flexShrink: 0,
                        }}
                      />
                      <Typography variant="body2" color="text.secondary">
                        {item}
                      </Typography>
                    </Box>
                  ))}
                </Stack>
              </AccordionDetails>
            </Accordion>
          ))}
        </Stack>
      </Box>
    </Box>
  );
};

export default StatisticsPanel; 