import React from 'react';
import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Box,
  List,
  ListItem,
  ListItemText,
  useTheme,
} from '@mui/material';
import {
  ExpandMore,
} from '@mui/icons-material';

interface AccordionSection {
  id: string;
  title: string;
  icon: React.ReactNode;
  iconColor: string;
  items: string[];
  description?: string;
}

interface InfoAccordionProps {
  sections: AccordionSection[];
  defaultExpanded?: string; // ID of section to expand by default
}

const InfoAccordion: React.FC<InfoAccordionProps> = ({ 
  sections, 
  defaultExpanded 
}) => {
  const theme = useTheme();

  return (
    <Box sx={{ width: '100%' }}>
      {sections.map((section) => (
        <Accordion
          key={section.id}
          defaultExpanded={section.id === defaultExpanded}
          sx={{
            mb: 2,
            border: '1px solid #E5E7EB',
            borderRadius: '12px !important',
            boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)',
            '&:before': {
              display: 'none', // Remove default MUI accordion divider
            },
            '&.Mui-expanded': {
              margin: '0 0 16px 0', // Maintain consistent spacing when expanded
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.12), 0 2px 4px rgba(0, 0, 0, 0.08)',
            },
          }}
        >
          <AccordionSummary
            expandIcon={
              <ExpandMore 
                sx={{ 
                  color: theme.palette.primary.main,
                  fontSize: 24,
                }} 
              />
            }
            sx={{
              px: 3,
              py: 2,
              minHeight: '72px',
              '&.Mui-expanded': {
                minHeight: '72px',
              },
              '& .MuiAccordionSummary-content': {
                margin: '12px 0',
                '&.Mui-expanded': {
                  margin: '12px 0',
                },
              },
            }}
          >
            <Box display="flex" alignItems="center" gap={3}>
              <Box
                sx={{
                  width: 40,
                  height: 40,
                  borderRadius: '10px',
                  backgroundColor: section.iconColor,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  boxShadow: `0 4px 6px ${section.iconColor}33`, // 20% opacity shadow
                }}
              >
                {section.icon}
              </Box>
              <Typography 
                variant="h6" 
                sx={{ 
                  fontWeight: 600,
                  color: 'text.primary',
                }}
              >
                {section.title}
              </Typography>
            </Box>
          </AccordionSummary>
          
          <AccordionDetails
            sx={{
              px: 3,
              pb: 3,
              pt: 0,
            }}
          >
            {section.description && (
              <Typography 
                variant="body1" 
                color="text.primary" 
                sx={{ mb: 2, fontWeight: 400 }}
              >
                {section.description}
              </Typography>
            )}
            
            <List sx={{ p: 0 }}>
              {section.items.map((item, index) => (
                <ListItem
                  key={index}
                  sx={{
                    px: 0,
                    py: 0.5,
                    alignItems: 'flex-start',
                  }}
                >
                  <Box
                    sx={{
                      width: 6,
                      height: 6,
                      borderRadius: '50%',
                      backgroundColor: theme.palette.primary.main,
                      mt: 1,
                      mr: 2,
                      flexShrink: 0,
                    }}
                  />
                  <ListItemText
                    primary={item}
                    primaryTypographyProps={{
                      variant: 'body1',
                      color: 'text.primary',
                      sx: { lineHeight: 1.6 },
                    }}
                  />
                </ListItem>
              ))}
            </List>
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );
};

export default InfoAccordion; 