Option Explicit

Dim docxPath, pdfPath, wordApp, wordDoc, pageCount
docxPath = WScript.Arguments(0)
pdfPath = WScript.Arguments(1)

On Error Resume Next
Set wordApp = CreateObject("Word.Application")
If Err.Number <> 0 Then
  WScript.Echo "CREATE_ERROR " & Err.Number & " " & Err.Description
  WScript.Quit 2
End If

wordApp.Visible = False
wordApp.DisplayAlerts = 0
Set wordDoc = wordApp.Documents.Open(docxPath, False, True)
If Err.Number <> 0 Then
  WScript.Echo "OPEN_ERROR " & Err.Number & " " & Err.Description
  wordApp.Quit
  WScript.Quit 3
End If

pageCount = wordDoc.ComputeStatistics(2)
WScript.Echo "PAGE_COUNT " & pageCount
wordDoc.ExportAsFixedFormat pdfPath, 17
If Err.Number <> 0 Then
  WScript.Echo "EXPORT_ERROR " & Err.Number & " " & Err.Description
  wordDoc.Close False
  wordApp.Quit
  WScript.Quit 4
End If

wordDoc.Close False
wordApp.Quit
WScript.Echo "PDF " & pdfPath
