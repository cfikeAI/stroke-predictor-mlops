{{- define "telemetry-api.fullname" -}}
{{-printf "%s-%s" .Chart.Name .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}