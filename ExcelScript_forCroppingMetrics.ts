function main(workbook: ExcelScript.Workbook) {
	let selectedSheet = workbook.getActiveWorksheet();
	let copyRange: ExcelScript.Range;
	let pasteRange: ExcelScript.Range;
	let row_val = 9
	let col_val = 1

	let pasteRow: number = row_val; // Adjust the row offset as needed
	let pasteColumn: number = col_val; // Adjust the column offset as needed

	for (let i = 16; i <= 750; i=i+5) {
	copyRange = selectedSheet.getRange(`A${i}:B${i}`);
	pasteRange = selectedSheet.getRange(`A${pasteRow}:B${pasteRow}`);
	copyRange.moveTo(pasteRange);
	col_val = col_val +2
	
	copyRange = selectedSheet.getRange(`A${i+1}:G${i+1}`);
	pasteRange = selectedSheet.getRange(`C${pasteRow}:I${pasteRow}`);
	copyRange.moveTo(pasteRange);
	col_val = col_val +7

	copyRange = selectedSheet.getRange(`A${i+2}:G${i+2}`);
	pasteRange = selectedSheet.getRange(`J${pasteRow}:P${pasteRow}`);
	copyRange.moveTo(pasteRange);
	col_val = col_val + 7

	copyRange = selectedSheet.getRange(`A${i+3}:G${i+3}`);
	pasteRange = selectedSheet.getRange(`Q${pasteRow}:W${pasteRow}`);
	copyRange.moveTo(pasteRange);
	col_val = col_val + 7

	copyRange = selectedSheet.getRange(`A${i+4}:G${i+4}`);
	pasteRange = selectedSheet.getRange(`X${pasteRow}:AD${pasteRow}`);
	copyRange.moveTo(pasteRange);
	col_val = col_val + 7


	row_val = row_val+1
	col_val = 1

	pasteRow = row_val; // Adjust the row offset as needed
	pasteColumn = col_val; // Adjust the column offset as needed


	}
}
