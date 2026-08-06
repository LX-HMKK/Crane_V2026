# 提交规范

提交信息按 Angular 规范写中文主题。在 PowerShell 中，避免把 `@` 解析成数组，使用 here-string 写入临时文件再提交：

```powershell
@'
fix(rdkx5): 修复导出脚本路径问题
- 修复 ONNX 路径使用 POSIX 斜杠
- 校准数据 pad 值对齐 Ultralytics
'@ | Out-File -Encoding utf8 .git-msg
git commit -F .git-msg; Remove-Item .git-msg
```

## 协作者

**禁止在提交信息中添加 `Co-Authored-By` 尾注或设置任何协作者。** 所有提交只记录实际执行该提交的作者。
