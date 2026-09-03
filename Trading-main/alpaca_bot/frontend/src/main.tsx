import React from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles/global.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: 180_000,
      refetchOnWindowFocus: true,
      staleTime: 30_000,
      retry: 0,
    },
  },
})

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = {
    error: null,
  }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('=== DASHBOARD CRASH ===')
    console.error(error)
    console.error(info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            minHeight: '100vh',
            background: '#0A0A0F',
            color: '#EAEAEA',
            padding: '40px',
            fontFamily: 'Arial, sans-serif',
          }}
        >
          <div
            style={{
              maxWidth: '900px',
              margin: '0 auto',
              padding: '24px',
              background: '#12121A',
              border: '1px solid #E61919',
              borderRadius: '8px',
            }}
          >
            <h1 style={{ color: '#E61919', marginBottom: '16px' }}>
              Dashboard Component Error
            </h1>

            <p style={{ marginBottom: '16px' }}>
              The application loaded, but a React component crashed.
            </p>

            <pre
              style={{
                background: '#08080C',
                color: '#FF6B6B',
                padding: '16px',
                borderRadius: '6px',
                whiteSpace: 'pre-wrap',
              }}
            >
              {this.state.error.message}
            </pre>

            <p
              style={{
                color: '#6B7280',
                marginTop: '16px',
                fontSize: '13px',
              }}
            >
              Press F12 and check the Console for the component name.
            </p>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

const rootElement = document.getElementById('root')

if (!rootElement) {
  throw new Error('Root element not found')
}

createRoot(rootElement).render(
  <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </ErrorBoundary>,
)
