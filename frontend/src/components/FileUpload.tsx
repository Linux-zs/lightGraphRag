import { useRef, useState } from 'react'

interface Props {
  onUpload: (file: File) => Promise<void>
  onMultiUpload?: (files: File[]) => Promise<void>
  accept?: string
  uploading?: boolean
}

const ACCEPT_TYPES = '.txt,.md,.pdf,.docx'

export default function FileUpload({ onUpload, onMultiUpload, accept = ACCEPT_TYPES, uploading }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState('')

  const handleFiles = async (files: FileList) => {
    setError('')
    const selectedFiles = Array.from(files).filter((file): file is File => file instanceof File)
    if (selectedFiles.length === 0) return
    try {
      if (selectedFiles.length > 1 && onMultiUpload) {
        await onMultiUpload(selectedFiles)
      } else {
        await onUpload(selectedFiles[0])
      }
    } catch (e: unknown) {
      setError((e as Error).message || 'Upload failed')
    }
  }

  return (
    <div>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          if (e.dataTransfer.files.length > 0) handleFiles(e.dataTransfer.files)
        }}
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors ${
          dragOver
            ? 'border-primary-400 bg-primary-50'
            : 'border-gray-300 bg-gray-50 hover:border-gray-400 hover:bg-gray-100'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) handleFiles(e.target.files)
            e.target.value = '' // reset so same file can be re-selected
          }}
        />
        {uploading ? (
          <div className="flex flex-col items-center gap-2">
            <div className="animate-spin w-6 h-6 border-2 border-primary-500 border-t-transparent rounded-full" />
            <span className="text-sm text-gray-500">正在上传文档...</span>
          </div>
        ) : (
          <>
            <span className="mx-auto mb-3 block h-8 w-6 rounded-[3px] border-2 border-gray-400 bg-white" />
            <p className="text-sm text-gray-600 font-medium">
              点击或拖拽文件到此处上传（支持多选）
            </p>
            <p className="text-xs text-gray-400 mt-1">
              支持 TXT, MD, PDF, DOCX，可一次选择多个文件
            </p>
          </>
        )}
      </div>
      {error && (
        <p className="mt-2 text-sm text-red-500">{error}</p>
      )}
    </div>
  )
}
