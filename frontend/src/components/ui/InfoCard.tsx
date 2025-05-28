import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  useTheme,
  alpha,
  Fade,
} from '@mui/material';

interface InfoCardProps {
  title: string;
  content: React.ReactNode;
  icon?: React.ReactNode;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info';
  variant?: 'outlined' | 'filled';
  className?: string;
}

const InfoCard: React.FC<InfoCardProps> = ({
  title,
  content,
  icon,
  color = 'primary',
  variant = 'outlined',
  className,
}) => {
  const theme = useTheme();

  const getCardStyles = () => {
    const colorPalette = theme.palette[color];
    
    if (variant === 'filled') {
      return {
        bgcolor: alpha(colorPalette.main, 0.05),
        border: `1px solid ${alpha(colorPalette.main, 0.2)}`,
        '&:hover': {
          bgcolor: alpha(colorPalette.main, 0.08),
          transform: 'translateY(-2px)',
          boxShadow: theme.shadows[4],
        },
      };
    }

    return {
      border: `1px solid ${theme.palette.divider}`,
      '&:hover': {
        borderColor: colorPalette.main,
        transform: 'translateY(-2px)',
        boxShadow: theme.shadows[4],
      },
    };
  };

  return (
    <Fade in timeout={300}>
      <Card
        className={className}
        sx={{
          ...getCardStyles(),
          borderRadius: 2,
          transition: 'all 0.3s ease',
          height: '100%',
        }}
      >
        <CardContent sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* Header */}
          <Box 
            display="flex" 
            alignItems="center" 
            gap={1} 
            mb={2}
          >
            {icon && (
              <Box 
                sx={{ 
                  color: theme.palette[color].main,
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                {icon}
              </Box>
            )}
            <Typography 
              variant="h6" 
              component="h3"
              sx={{ 
                color: theme.palette[color].main,
                fontWeight: 600,
              }}
            >
              {title}
            </Typography>
          </Box>

          {/* Content */}
          <Box sx={{ flex: 1 }}>
            {typeof content === 'string' ? (
              <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                {content}
              </Typography>
            ) : (
              content
            )}
          </Box>
        </CardContent>
      </Card>
    </Fade>
  );
};

export default InfoCard; 