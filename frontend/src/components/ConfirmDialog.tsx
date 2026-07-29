import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react'
import { AlertTriangle, X } from 'lucide-react'

export interface ConfirmOptions {
  title: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  tone?: 'default' | 'danger'
}

type ConfirmFn = (options: ConfirmOptions) => Promise<boolean>

const ConfirmContext = createContext<ConfirmFn | null>(null)

export function ConfirmProvider({ children }: { children: ReactNode }) {
  const [options, setOptions] = useState<ConfirmOptions | null>(null)
  const resolverRef = useRef<((confirmed: boolean) => void) | null>(null)

  const close = useCallback((confirmed: boolean) => {
    resolverRef.current?.(confirmed)
    resolverRef.current = null
    setOptions(null)
  }, [])

  const confirm = useCallback<ConfirmFn>((nextOptions) => {
    resolverRef.current?.(false)
    setOptions(nextOptions)
    return new Promise<boolean>((resolve) => {
      resolverRef.current = resolve
    })
  }, [])

  useEffect(() => {
    if (!options) return
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') close(false)
    }
    document.addEventListener('keydown', handleEscape)
    return () => document.removeEventListener('keydown', handleEscape)
  }, [close, options])

  return (
    <ConfirmContext.Provider value={confirm}>
      {children}
      {options && (
        <div
          className="fixed inset-0 z-[120] flex items-center justify-center bg-gray-950/35 p-4 backdrop-blur-[1px]"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) close(false)
          }}
        >
          <div
            className="w-full max-w-md rounded-lg border border-gray-200 bg-white shadow-2xl"
            role="alertdialog"
            aria-modal="true"
            aria-labelledby="confirm-dialog-title"
            aria-describedby="confirm-dialog-message"
          >
            <div className="flex items-start gap-3 px-5 py-5">
              <span className={`grid h-9 w-9 shrink-0 place-items-center rounded-lg ${
                options.tone === 'danger'
                  ? 'bg-red-50 text-red-600'
                  : 'bg-amber-50 text-amber-600'
              }`}>
                <AlertTriangle size={18} />
              </span>
              <div className="min-w-0 flex-1">
                <h2 id="confirm-dialog-title" className="text-base font-semibold text-gray-900">
                  {options.title}
                </h2>
                <p id="confirm-dialog-message" className="mt-2 text-sm leading-relaxed text-gray-600">
                  {options.message}
                </p>
              </div>
              <button
                type="button"
                onClick={() => close(false)}
                className="ui-icon-button -mr-1 -mt-1"
                aria-label="关闭"
              >
                <X size={16} />
              </button>
            </div>
            <div className="flex justify-end gap-2 border-t border-gray-100 px-5 py-3">
              <button type="button" onClick={() => close(false)} className="ui-button-secondary">
                {options.cancelLabel || '取消'}
              </button>
              <button
                type="button"
                onClick={() => close(true)}
                className={options.tone === 'danger'
                  ? 'ui-button border border-red-600 bg-red-600 text-white hover:bg-red-700'
                  : 'ui-button-primary'}
                autoFocus
              >
                {options.confirmLabel || '确认'}
              </button>
            </div>
          </div>
        </div>
      )}
    </ConfirmContext.Provider>
  )
}

export function useConfirm() {
  const confirm = useContext(ConfirmContext)
  if (!confirm) throw new Error('useConfirm must be used inside ConfirmProvider')
  return confirm
}
