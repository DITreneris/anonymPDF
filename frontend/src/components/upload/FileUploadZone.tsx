import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import type { FileRejection } from 'react-dropzone';
import {
  Box,
  Typography,
  Alert,
  Chip,
  Stack,
  useTheme,
  alpha,
  Fade,
  Zoom,
} from '@mui/material';
import {
  CloudUpload,
  CheckCircle,
  Error as ErrorIcon,
  Description,
} from '@mui/icons-material';

interface FileUploadZoneProps {
  onFileAccepted: (file: File) => void;
  onFileRejected?: (rejections: FileRejection[]) => void;
  disabled?: boolean;
  maxSize?: number; // in bytes, default 50MB
  currentFileName?: string | null;
  isProcessing?: boolean;
}

const FileUploadZone: React.FC<FileUploadZoneProps> = ({
  onFileAccepted,
  onFileRejected,
  disabled = false,
  maxSize = 50 * 1024 * 1024, // 50MB
  currentFileName = null,
  isProcessing = false,
}) => {
  const theme = useTheme();
  const [dragError, setDragError] = useState<string | null>(null);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const validateFile = useCallback((file: File): string | null => {
    // Check file type
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      return 'Only PDF files are supported';
    }

    // Check file size
    if (file.size > maxSize) {
      return `File size must be less than ${formatFileSize(maxSize)}`;
    }

    // Check if file is empty
    if (file.size === 0) {
      return 'File appears to be empty';
    }

    return null;
  }, [maxSize]);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
    setDragError(null);

    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0];
      let errorMessage = 'File upload failed';

      if (rejection.errors.length > 0) {
        const error = rejection.errors[0];
        switch (error.code) {
          case 'file-too-large':
            errorMessage = `File is too large. Maximum size is ${formatFileSize(maxSize)}`;
            break;
          case 'file-invalid-type':
            errorMessage = 'Only PDF files are supported';
            break;
          default:
            errorMessage = error.message;
        }
      }

      setDragError(errorMessage);
      onFileRejected?.(rejectedFiles);
      return;
    }

    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      const validationError = validateFile(file);

      if (validationError) {
        setDragError(validationError);
        return;
      }

      onFileAccepted(file);
    }
  }, [onFileAccepted, onFileRejected, validateFile, maxSize]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false,
    disabled: disabled || isProcessing,
    maxSize,
  });

  const getUploadZoneStyles = () => {
    const baseStyles = {
      p: 4,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      cursor: disabled || isProcessing ? 'not-allowed' : 'pointer',
      border: '2px dashed',
      borderRadius: 3,
      transition: 'all 0.3s ease',
      position: 'relative',
      minHeight: '200px',
      justifyContent: 'center',
      textAlign: 'center',
    };

    if (disabled || isProcessing) {
      return {
        ...baseStyles,
        borderColor: theme.palette.action.disabled,
        bgcolor: alpha(theme.palette.action.disabled, 0.05),
        color: theme.palette.action.disabled,
      };
    }

    if (isDragReject || dragError) {
      return {
        ...baseStyles,
        borderColor: theme.palette.error.main,
        bgcolor: alpha(theme.palette.error.main, 0.05),
        color: theme.palette.error.main,
        transform: 'scale(0.98)',
      };
    }

    if (isDragActive) {
      return {
        ...baseStyles,
        borderColor: theme.palette.primary.main,
        bgcolor: alpha(theme.palette.primary.main, 0.08),
        color: theme.palette.primary.main,
        transform: 'scale(1.02)',
        boxShadow: `0 0 20px ${alpha(theme.palette.primary.main, 0.3)}`,
      };
    }

    if (currentFileName) {
      return {
        ...baseStyles,
        borderColor: theme.palette.success.main,
        bgcolor: alpha(theme.palette.success.main, 0.05),
        color: theme.palette.success.main,
      };
    }

    return {
      ...baseStyles,
      borderColor: theme.palette.grey[300],
      bgcolor: theme.palette.background.paper,
      color: theme.palette.text.secondary,
      '&:hover': {
        borderColor: theme.palette.primary.main,
        bgcolor: alpha(theme.palette.primary.main, 0.02),
        transform: 'translateY(-2px)',
        boxShadow: theme.shadows[4],
      },
    };
  };

  const getIconComponent = () => {
    if (dragError || isDragReject) {
      return <ErrorIcon sx={{ fontSize: 64, mb: 2 }} />;
    }
    if (currentFileName) {
      return <CheckCircle sx={{ fontSize: 64, mb: 2 }} />;
    }
    return <CloudUpload sx={{ fontSize: 64, mb: 2 }} />;
  };

  const getMainText = () => {
    if (disabled) return 'Upload disabled';
    if (isProcessing) return 'Processing...';
    if (dragError) return 'Upload Error';
    if (isDragReject) return 'Invalid file type';
    if (isDragActive) return 'Drop the PDF here';
    if (currentFileName) return 'File ready for processing';
    return 'Drag & drop your PDF';
  };

  const getSubText = () => {
    if (disabled || isProcessing) return '';
    if (dragError) return dragError;
    if (isDragReject) return 'Only PDF files are supported';
    if (isDragActive) return 'Release to upload';
    if (currentFileName) return `Selected: ${currentFileName}`;
    return 'or click to select a file';
  };

  return (
    <Box>
      <Box
        {...getRootProps()}
        sx={getUploadZoneStyles()}
        role="button"
        tabIndex={disabled || isProcessing ? -1 : 0}
        aria-label="File upload area"
        aria-describedby="upload-instructions"
      >
        <input {...getInputProps()} aria-label="File input" />
        
        <Zoom in timeout={300}>
          {getIconComponent()}
        </Zoom>

        <Typography 
          variant="h6" 
          gutterBottom
          sx={{ 
            fontWeight: 600,
            mb: 1,
          }}
        >
          {getMainText()}
        </Typography>

        <Typography 
          variant="body2" 
          sx={{ 
            opacity: 0.8,
            mb: 2,
          }}
          id="upload-instructions"
        >
          {getSubText()}
        </Typography>

        {!disabled && !isProcessing && !currentFileName && (
          <Fade in timeout={500}>
            <Stack direction="row" spacing={1} flexWrap="wrap" justifyContent="center">
              <Chip 
                label={`Max ${formatFileSize(maxSize)}`}
                size="small"
                variant="outlined"
                sx={{ opacity: 0.7 }}
              />
              <Chip 
                label="PDF only"
                size="small"
                variant="outlined"
                sx={{ opacity: 0.7 }}
              />
            </Stack>
          </Fade>
        )}

        {currentFileName && (
          <Fade in timeout={300}>
            <Box
              sx={{
                mt: 2,
                p: 2,
                bgcolor: alpha(theme.palette.success.main, 0.1),
                borderRadius: 2,
                border: `1px solid ${alpha(theme.palette.success.main, 0.3)}`,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}
            >
              <Description color="success" />
              <Typography variant="body2" color="success.main">
                {currentFileName}
              </Typography>
            </Box>
          </Fade>
        )}
      </Box>

      {dragError && (
        <Fade in>
          <Alert 
            severity="error" 
            sx={{ mt: 2 }}
            onClose={() => setDragError(null)}
          >
            {dragError}
          </Alert>
        </Fade>
      )}
    </Box>
  );
};

export default FileUploadZone; 