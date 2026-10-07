import { useEffect, useState } from 'react'
import { AdminPage } from './components/Admin/AdminPage'
import { ChatShell } from './components/Chat/ChatShell'
import { ErrorBoundary } from './components/ErrorBoundary'

export default function App() {
  const [hash, setHash] = useState(window.location.hash)

  useEffect(() => {
    const onHashChange = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  return (
    <ErrorBoundary fallbackLabel="Không thể tải giao diện. Hãy thử lại.">
      {hash.startsWith('#admin') ? <AdminPage /> : <ChatShell />}
    </ErrorBoundary>
  )
}
