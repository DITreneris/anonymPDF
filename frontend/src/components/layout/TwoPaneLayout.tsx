import React from 'react';
import {
  Box,
  Paper,
  useTheme,
  useMediaQuery,
  Fade,
} from '@mui/material';

interface TwoPaneLayoutProps {
  leftPane: React.ReactNode;
  rightPane: React.ReactNode;
  leftPaneWidth?: number; // Percentage width for left pane (default: 50)
  spacing?: number; // Spacing between panes (default: 3)
  elevation?: number; // Paper elevation (default: 1)
  className?: string;
}

const TwoPaneLayout: React.FC<TwoPaneLayoutProps> = ({
  leftPane,
  rightPane,
  leftPaneWidth = 50,
  spacing = 4,
  elevation = 0,
  className,
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  return (
    <Box
      className={className}
      sx={{
        width: '100%',
        height: '100%',
        minHeight: isMobile ? 'auto' : '600px',
        display: 'flex',
        flexDirection: isMobile ? 'column' : 'row',
        gap: spacing,
      }}
    >
      {/* Left Pane - Upload Controls & Status */}
      <Box
        sx={{
          width: isMobile ? '100%' : `${leftPaneWidth}%`,
          height: isMobile ? 'auto' : '100%',
          minHeight: isMobile ? '400px' : '600px',
        }}
      >
        <Fade in timeout={300}>
          <Paper
            elevation={elevation}
            sx={{
              height: '100%',
              p: 4, // 32px padding (4 * 8px)
              display: 'flex',
              flexDirection: 'column',
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: '12px',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
              transition: 'all 0.2s ease',
              '&:hover': {
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.12), 0 2px 4px rgba(0, 0, 0, 0.08)',
              },
            }}
          >
            {leftPane}
          </Paper>
        </Fade>
      </Box>

      {/* Right Pane - Preview, Results, Help */}
      <Box
        sx={{
          width: isMobile ? '100%' : `${100 - leftPaneWidth}%`,
          height: isMobile ? 'auto' : '100%',
          minHeight: isMobile ? '300px' : '600px',
        }}
      >
        <Fade in timeout={500}>
          <Paper
            elevation={elevation}
            sx={{
              height: '100%',
              p: 4, // 32px padding (4 * 8px)
              display: 'flex',
              flexDirection: 'column',
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: '12px',
              boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
              transition: 'all 0.2s ease',
              position: 'relative',
              overflow: 'hidden',
              '&:hover': {
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.12), 0 2px 4px rgba(0, 0, 0, 0.08)',
              },
              // Subtle accent line at top - more visible
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '3px',
                background: theme.palette.primary.main,
                opacity: 0.15,
              },
            }}
          >
            {rightPane}
          </Paper>
        </Fade>
      </Box>
    </Box>
  );
};

export default TwoPaneLayout; 