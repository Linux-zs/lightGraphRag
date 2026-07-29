import type { InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from 'react'

export function Panel({
  children,
  className = '',
}: {
  children: ReactNode
  className?: string
}) {
  return <section className={`ui-panel ${className}`}>{children}</section>
}

export function SelectField({
  label,
  className = '',
  children,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement> & {
  label?: string
  children: ReactNode
}) {
  return (
    <label className={`block min-w-0 ${className}`}>
      {label && <span className="ui-label">{label}</span>}
      <select {...props} className="ui-control w-full">
        {children}
      </select>
    </label>
  )
}

export function Toggle({
  checked,
  onChange,
  label,
  disabled = false,
}: {
  checked: boolean
  onChange: (checked: boolean) => void
  label: string
  disabled?: boolean
}) {
  return (
    <label className={`inline-flex items-center gap-2 text-sm ${disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
        className="peer sr-only"
      />
      <span className="relative h-5 w-9 shrink-0 rounded-full bg-gray-300 transition-colors after:absolute after:left-0.5 after:top-0.5 after:h-4 after:w-4 after:rounded-full after:bg-white after:shadow-sm after:transition-transform peer-checked:bg-primary-600 peer-checked:after:translate-x-4 peer-focus-visible:ring-2 peer-focus-visible:ring-primary-300 peer-focus-visible:ring-offset-2" />
      <span className="text-gray-700">{label}</span>
    </label>
  )
}

export function RangeField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
}) {
  return (
    <label className="block min-w-0">
      <span className="ui-label">{label}</span>
      <div className="flex items-center gap-3">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          className="ui-range min-w-0 flex-1"
        />
        <input
          type="number"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => {
            const next = Number(event.target.value)
            if (Number.isFinite(next)) onChange(Math.min(max, Math.max(min, next)))
          }}
          className="ui-control w-20 px-2 text-right tabular-nums"
        />
      </div>
    </label>
  )
}

export function TextInput({
  label,
  className = '',
  ...props
}: InputHTMLAttributes<HTMLInputElement> & { label?: string }) {
  return (
    <label className={`block min-w-0 ${className}`}>
      {label && <span className="ui-label">{label}</span>}
      <input {...props} className="ui-control w-full" />
    </label>
  )
}
