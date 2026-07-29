interface Props {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  unit?: string
  hint?: string
}

export default function ScoreSlider({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.05,
  unit = '',
  hint,
}: Props) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium text-gray-700">{label}</label>
        <span className="text-sm text-primary-600 font-mono tabular-nums">
          {value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 bg-gray-200 rounded-full appearance-none cursor-pointer accent-primary-500"
      />
      {hint && <p className="text-[11px] text-gray-400">{hint}</p>}
    </div>
  )
}
