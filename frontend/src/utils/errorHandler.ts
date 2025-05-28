import { type ProcessingError } from '../components/ErrorDialog';

export interface ApiError {
  message: string;
  status?: number;
  code?: string;
  details?: any;
}

/**
 * Converts various error types to our structured ProcessingError format
 */
export const createProcessingError = (
  error: any,
  context?: Record<string, any>
): ProcessingError => {
  const timestamp = new Date().toISOString();

  // Handle Axios errors
  if (error.response) {
    const status = error.response.status;
    const data = error.response.data;
    
    // Determine error type based on status code
    let type: ProcessingError['type'] = 'system';
    let message = 'An error occurred while processing your request';
    let details = '';
    let recoveryActions: string[] = [];

    if (status >= 400 && status < 500) {
      // Client errors
      type = 'validation';
      
      if (status === 400) {
        message = 'Invalid request';
        details = data?.detail || 'The request was not valid. Please check your input.';
        recoveryActions = [
          'Check that your file is a valid PDF',
          'Ensure the file is not corrupted or password-protected',
          'Try with a different file'
        ];
      } else if (status === 413) {
        message = 'File too large';
        details = 'The uploaded file exceeds the maximum size limit.';
        recoveryActions = [
          'Try with a smaller PDF file',
          'Compress your PDF before uploading',
          'Contact support for large file processing'
        ];
      } else if (status === 415) {
        message = 'Unsupported file type';
        details = 'Only PDF files are supported.';
        recoveryActions = [
          'Ensure your file has a .pdf extension',
          'Convert your document to PDF format',
          'Check that the file is not corrupted'
        ];
      } else {
        message = data?.detail || 'Request validation failed';
        details = typeof data === 'string' ? data : JSON.stringify(data);
      }
    } else if (status >= 500) {
      // Server errors
      type = 'system';
      message = 'Server error occurred';
      details = data?.detail || 'An internal server error occurred. Please try again.';
      recoveryActions = [
        'Wait a moment and try again',
        'Check if the service is available',
        'Contact support if the problem persists'
      ];
    }

    return {
      type,
      message,
      details,
      recoveryActions,
      errorCode: `HTTP_${status}`,
      timestamp,
      context: {
        status,
        url: error.config?.url,
        method: error.config?.method,
        ...context
      }
    };
  }

  // Handle network errors
  if (error.request) {
    return {
      type: 'network',
      message: 'Network connection failed',
      details: 'Unable to connect to the server. Please check your internet connection.',
      recoveryActions: [
        'Check your internet connection',
        'Try again in a few moments',
        'Refresh the page if the problem persists',
        'Contact support if you continue to have connectivity issues'
      ],
      errorCode: 'NETWORK_ERROR',
      timestamp,
      context: {
        url: error.config?.url,
        ...context
      }
    };
  }

  // Handle timeout errors
  if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
    return {
      type: 'timeout',
      message: 'Request timed out',
      details: 'The operation took too long to complete.',
      recoveryActions: [
        'Try again with a smaller file',
        'Check your internet connection speed',
        'Contact support for processing large files'
      ],
      errorCode: 'TIMEOUT_ERROR',
      timestamp,
      context
    };
  }

  // Handle processing errors from our backend
  if (error.message && typeof error.message === 'string') {
    try {
      const parsed = JSON.parse(error.message);
      if (parsed.error && parsed.details) {
        return {
          type: 'processing',
          message: parsed.error,
          details: parsed.details,
          recoveryActions: [
            'Try uploading the file again',
            'Check if the PDF contains readable text',
            'Ensure the PDF is not password-protected',
            'Contact support if the problem persists'
          ],
          errorCode: 'PROCESSING_ERROR',
          timestamp,
          context: {
            originalError: parsed,
            ...context
          }
        };
      }
    } catch {
      // Not JSON, treat as regular error
    }
  }

  // Handle generic JavaScript errors
  if (error instanceof Error) {
    return {
      type: 'system',
      message: error.message || 'An unexpected error occurred',
      details: error.stack || 'No additional details available',
      recoveryActions: [
        'Refresh the page and try again',
        'Clear your browser cache',
        'Contact support if the issue persists'
      ],
      errorCode: 'JS_ERROR',
      timestamp,
      context: {
        errorName: error.name,
        ...context
      }
    };
  }

  // Fallback for unknown error types
  return {
    type: 'system',
    message: 'An unknown error occurred',
    details: typeof error === 'string' ? error : JSON.stringify(error),
    recoveryActions: [
      'Try the operation again',
      'Refresh the page',
      'Contact support if needed'
    ],
    errorCode: 'UNKNOWN_ERROR',
    timestamp,
    context: {
      originalError: error,
      ...context
    }
  };
};

/**
 * Handles errors during file upload with specific context
 */
export const handleUploadError = (error: any, fileName?: string): ProcessingError => {
  return createProcessingError(error, {
    operation: 'file_upload',
    fileName
  });
};

/**
 * Handles errors during PDF processing with specific context
 */
export const handleProcessingError = (error: any, documentId?: string): ProcessingError => {
  return createProcessingError(error, {
    operation: 'pdf_processing',
    documentId
  });
};

/**
 * Handles errors during report fetching with specific context
 */
export const handleReportError = (error: any, documentId?: string): ProcessingError => {
  return createProcessingError(error, {
    operation: 'report_fetch',
    documentId
  });
};

/**
 * Logs errors to console in development and potentially to error tracking service in production
 */
export const logError = (error: ProcessingError, additionalContext?: Record<string, any>) => {
  const errorData = {
    ...error,
    additionalContext,
    userAgent: navigator.userAgent,
    url: window.location.href,
    timestamp: new Date().toISOString()
  };

  // Log to console in development
  if (import.meta.env.DEV) {
    console.error('Error logged:', errorData);
  }

  // In production, you would send this to your error tracking service
  // Example: errorTrackingService.captureException(error, { extra: errorData });
};

/**
 * Determines if an error is retryable based on its type
 */
export const isRetryableError = (error: ProcessingError): boolean => {
  return ['network', 'timeout', 'system'].includes(error.type);
};

/**
 * Gets user-friendly error message for display
 */
export const getDisplayMessage = (error: ProcessingError): string => {
  switch (error.type) {
    case 'validation':
      return 'Please check your file and try again';
    case 'processing':
      return 'There was an issue processing your PDF';
    case 'network':
      return 'Connection issue - please check your internet';
    case 'timeout':
      return 'The operation timed out - please try again';
    case 'system':
      return 'A system error occurred - please try again';
    default:
      return error.message;
  }
}; 