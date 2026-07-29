import { useMemo, useState } from 'react'
import { Check, ChevronDown } from 'lucide-react'

interface Props {
  value: string
  options: string[]
  onChange: (value: string) => void
  placeholder?: string
}

export default function ModelCombobox({ value, options, onChange, placeholder }: Props) {
  const [open, setOpen] = useState(false)
  const filtered = useMemo(() => {
    const needle = value.trim().toLowerCase()
    if (!needle) return options
    const matches = options.filter((item) => item.toLowerCase().includes(needle))
    return matches.length > 0 ? matches : options
  }, [options, value])

  return (
    <div className="relative min-w-0">
      <div className="flex h-10 rounded-md border border-gray-300 bg-white focus-within:border-primary-500 focus-within:ring-2 focus-within:ring-primary-100">
        <input
          value={value}
          onChange={(event) => {
            onChange(event.target.value)
            setOpen(true)
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => window.setTimeout(() => setOpen(false), 120)}
          placeholder={placeholder || '选择或输入模型名称'}
          className="min-w-0 flex-1 bg-transparent px-3 text-sm outline-none"
          autoComplete="off"
        />
        <button
          type="button"
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => setOpen((current) => !current)}
          className="grid w-9 place-items-center text-gray-400 hover:text-gray-700"
          title="展开模型列表"
        >
          <ChevronDown size={16} className={open ? 'rotate-180 transition-transform' : 'transition-transform'} />
        </button>
      </div>
      {open && options.length > 0 && (
        <div className="absolute z-30 mt-1 max-h-60 w-full overflow-y-auto rounded-md border border-gray-200 bg-white p-1 shadow-lg">
          {filtered.map((item) => (
            <button
              key={item}
              type="button"
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => {
                onChange(item)
                setOpen(false)
              }}
              className="flex w-full items-center gap-2 rounded px-2.5 py-2 text-left text-sm text-gray-700 hover:bg-gray-100"
            >
              <span className="min-w-0 flex-1 truncate">{item}</span>
              {item === value && <Check size={14} className="shrink-0 text-primary-600" />}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
