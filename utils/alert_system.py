import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging


@dataclass
class AlertInfo:
    """
    Información de una alerta de seguridad
    """
    person_id: int
    person_name: str
    person_lastname: str
    student_id: Optional[str]
    requisition_type: str
    confidence: float
    detection_timestamp: str
    image_path: str
    alert_level: str  # "HIGH", "MEDIUM", "LOW"
    location: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class AlertSystem:
    """
    Sistema de alertas para personas requisitoriadas
    """

    def __init__(self, alerts_log_path: str = "storage/logs/security_alerts.json"):
        """
        Inicializa el sistema de alertas

        Args:
            alerts_log_path: Ruta del archivo de log de alertas
        """
        self.alerts_log_path = alerts_log_path
        self.setup_logging()

        # Crear directorio si no existe
        os.makedirs(os.path.dirname(alerts_log_path), exist_ok=True)

        # Configuración de alertas
        self.alert_thresholds = {
            "HIGH": 80.0,  # Confianza >= 80%
            "MEDIUM": 60.0,  # Confianza >= 60%
            "LOW": 40.0  # Confianza >= 40%
        }

        # Tipos de requisitorias y sus niveles de alerta
        self.requisition_alert_levels = {
            "Hurto": "MEDIUM",
            "Robo": "HIGH",
            "Estafa": "MEDIUM",
            "Vandalismo": "LOW",
            "Disturbios": "MEDIUM",
            "Violencia doméstica": "HIGH",
            "Fraude": "MEDIUM",
            "Tráfico": "HIGH",
            "Falsificación": "MEDIUM",
            "Agresión": "HIGH",
            "Amenazas": "HIGH",
            "Violación de medidas cautelares": "HIGH"
        }

    def setup_logging(self):
        """
        Configura el sistema de logging
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('storage/logs/alert_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def generate_security_alert(self, alert_info: AlertInfo) -> Dict[str, Any]:
        """
        Genera una alerta de seguridad

        Args:
            alert_info: Información de la alerta

        Returns:
            Diccionario con la respuesta de la alerta
        """
        # Determinar nivel de alerta
        alert_level = self._determine_alert_level(alert_info.confidence, alert_info.requisition_type)
        alert_info.alert_level = alert_level

        # Crear mensaje de alerta
        alert_message = self._create_alert_message(alert_info)

        # Registrar alerta
        alert_record = self._log_alert(alert_info)

        # Simular notificación a autoridades
        notification_result = self._simulate_authority_notification(alert_info)

        # Respuesta completa
        response = {
            "alert_generated": True,
            "alert_id": alert_record["alert_id"],
            "alert_level": alert_level,
            "message": alert_message,
            "person_info": {
                "id": alert_info.person_id,
                "name": f"{alert_info.person_name} {alert_info.person_lastname}",
                "student_id": alert_info.student_id,
                "requisition_type": alert_info.requisition_type
            },
            "detection_info": {
                "confidence": alert_info.confidence,
                "timestamp": alert_info.detection_timestamp,
                "location": alert_info.location
            },
            "authority_notification": notification_result,
            "recommended_actions": self._get_recommended_actions(alert_level, alert_info.requisition_type),
            "timestamp": datetime.now().isoformat()
        }

        # Log de la alerta
        self.logger.warning(
            f"🚨 ALERTA DE SEGURIDAD - {alert_level} - Persona requisitoriada detectada: {alert_info.person_name} {alert_info.person_lastname} (ID: {alert_info.person_id})")

        return response

    def _determine_alert_level(self, confidence: float, requisition_type: str) -> str:
        """
        Determina el nivel de alerta basado en confianza y tipo de requisitoria
        """
        # Nivel base por confianza
        base_level = "LOW"
        if confidence >= self.alert_thresholds["HIGH"]:
            base_level = "HIGH"
        elif confidence >= self.alert_thresholds["MEDIUM"]:
            base_level = "MEDIUM"

        # Ajustar por tipo de requisitoria
        requisition_level = self.requisition_alert_levels.get(requisition_type, "MEDIUM")

        # Tomar el nivel más alto
        levels_priority = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

        final_level_priority = max(
            levels_priority.get(base_level, 1),
            levels_priority.get(requisition_level, 2)
        )

        for level, priority in levels_priority.items():
            if priority == final_level_priority:
                return level

        return "MEDIUM"

    def _create_alert_message(self, alert_info: AlertInfo) -> str:
        """
        Crea el mensaje de alerta
        """
        messages = {
            "HIGH": "🚨 ¡ALERTA CRÍTICA DE SEGURIDAD! 🚨",
            "MEDIUM": "⚠️ ALERTA DE SEGURIDAD ⚠️",
            "LOW": "🔔 Notificación de Seguridad 🔔"
        }

        base_message = messages.get(alert_info.alert_level, messages["MEDIUM"])

        detailed_message = f"""
        {base_message}

        PERSONA REQUISITORIADA DETECTADA

        👤 Información Personal:
        - Nombre: {alert_info.person_name} {alert_info.person_lastname}
        - ID Persona: {alert_info.person_id}
        - ID Estudiante: {alert_info.student_id or 'N/A'}

        ⚖️ Información Legal:
        - Tipo de Requisitoria: {alert_info.requisition_type}
        - Nivel de Alerta: {alert_info.alert_level}

        🔍 Información de Detección:
        - Confianza: {alert_info.confidence:.1f}%
        - Fecha/Hora: {alert_info.detection_timestamp}
        - Ubicación: {alert_info.location or 'No especificada'}

        🚔 NOTIFICACIÓN ENVIADA A LAS AUTORIDADES (SIMULADA)

        ⚡ ACCIÓN INMEDIATA REQUERIDA
        """

        return detailed_message.strip()

    def _log_alert(self, alert_info: AlertInfo) -> Dict[str, Any]:
        """
        Registra la alerta en el log
        """
        alert_id = f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{alert_info.person_id}"

        alert_record = {
            "alert_id": alert_id,
            "alert_info": asdict(alert_info),
            "logged_at": datetime.now().isoformat(),
            "status": "ACTIVE"
        }

        # Leer alertas existentes
        existing_alerts = []
        if os.path.exists(self.alerts_log_path):
            try:
                with open(self.alerts_log_path, 'r', encoding='utf-8') as f:
                    existing_alerts = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_alerts = []

        # Añadir nueva alerta
        existing_alerts.append(alert_record)

        # Guardar alertas actualizadas
        with open(self.alerts_log_path, 'w', encoding='utf-8') as f:
            json.dump(existing_alerts, f, indent=2, ensure_ascii=False)

        return alert_record

    def _simulate_authority_notification(self, alert_info: AlertInfo) -> Dict[str, Any]:
        """
        Simula la notificación a las autoridades
        """
        notification_methods = []

        if alert_info.alert_level == "HIGH":
            notification_methods = [
                "Policía Nacional - Central de Emergencias 911",
                "Seguridad del Campus - Línea Directa",
                "Supervisor de Seguridad - Notificación SMS",
                "Sistema de Videovigilancia - Alerta Automática"
            ]
        elif alert_info.alert_level == "MEDIUM":
            notification_methods = [
                "Seguridad del Campus - Línea Directa",
                "Supervisor de Seguridad - Notificación SMS"
            ]
        else:  # LOW
            notification_methods = [
                "Supervisor de Seguridad - Notificación Email"
            ]

        return {
            "status": "SIMULADO - EXITOSO",
            "notification_methods": notification_methods,
            "estimated_response_time": self._get_estimated_response_time(alert_info.alert_level),
            "reference_number": f"REF-{datetime.now().strftime('%Y%m%d%H%M%S')}-{alert_info.person_id}",
            "note": "ESTA ES UNA SIMULACIÓN - EN PRODUCCIÓN SE INTEGRARÍA CON SISTEMAS REALES DE EMERGENCIA"
        }

    def _get_estimated_response_time(self, alert_level: str) -> str:
        """
        Obtiene el tiempo estimado de respuesta según el nivel de alerta
        """
        response_times = {
            "HIGH": "2-5 minutos",
            "MEDIUM": "5-15 minutos",
            "LOW": "15-30 minutos"
        }
        return response_times.get(alert_level, "15-30 minutos")

    def _get_recommended_actions(self, alert_level: str, requisition_type: str) -> List[str]:
        """
        Obtiene acciones recomendadas según el nivel de alerta
        """
        base_actions = {
            "HIGH": [
                "Contactar inmediatamente a la policía",
                "Activar protocolo de seguridad nivel crítico",
                "Evacuar área circundante si es necesario",
                "Documentar toda la evidencia",
                "Mantener vigilancia continua hasta llegada de autoridades"
            ],
            "MEDIUM": [
                "Notificar a seguridad del campus",
                "Incrementar vigilancia en el área",
                "Documentar el incidente",
                "Preparar información para autoridades"
            ],
            "LOW": [
                "Registrar el incidente",
                "Mantener vigilancia discreta",
                "Reportar a supervisor de seguridad"
            ]
        }

        # Acciones específicas por tipo de requisitoria
        specific_actions = {
            "Robo": ["Verificar inventario de bienes", "Revisar cámaras de seguridad adicionales"],
            "Violencia doméstica": ["Contactar servicios de apoyo a víctimas", "Protocolo de protección"],
            "Tráfico": ["Contactar unidad antinarcóticos", "Revisar áreas de alta vulnerabilidad"],
            "Agresión": ["Protocolo médico de emergencia", "Seguridad reforzada"]
        }

        actions = base_actions.get(alert_level, base_actions["MEDIUM"]).copy()

        if requisition_type in specific_actions:
            actions.extend(specific_actions[requisition_type])

        return actions

    def get_alert_history(self, limit: int = 50, alert_level: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtiene el historial de alertas

        Args:
            limit: Número máximo de alertas a retornar
            alert_level: Filtrar por nivel de alerta (opcional)

        Returns:
            Lista de alertas
        """
        if not os.path.exists(self.alerts_log_path):
            return []

        try:
            with open(self.alerts_log_path, 'r', encoding='utf-8') as f:
                alerts = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

        # Filtrar por nivel de alerta si se especifica
        if alert_level:
            alerts = [alert for alert in alerts
                      if alert.get('alert_info', {}).get('alert_level') == alert_level]

        # Ordenar por fecha (más recientes primero)
        alerts.sort(key=lambda x: x.get('logged_at', ''), reverse=True)

        return alerts[:limit]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de las alertas
        """
        alerts = self.get_alert_history(limit=1000)  # Obtener todas las alertas

        if not alerts:
            return {
                "total_alerts": 0,
                "by_level": {"HIGH": 0, "MEDIUM": 0, "LOW": 0},
                "by_requisition_type": {},
                "daily_average": 0,
                "most_common_requisition": None
            }

        # Contar por nivel
        level_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        requisition_counts = {}

        for alert in alerts:
            alert_info = alert.get('alert_info', {})
            level = alert_info.get('alert_level', 'UNKNOWN')
            requisition_type = alert_info.get('requisition_type', 'UNKNOWN')

            if level in level_counts:
                level_counts[level] += 1

            requisition_counts[requisition_type] = requisition_counts.get(requisition_type, 0) + 1

        # Calcular promedio diario (últimos 30 días)
        from datetime import datetime, timedelta
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_alerts = [
            alert for alert in alerts
            if datetime.fromisoformat(alert.get('logged_at', '').replace('Z', '+00:00')) >= thirty_days_ago
        ]
        daily_average = len(recent_alerts) / 30

        # Tipo de requisitoria más común
        most_common_requisition = max(requisition_counts, key=requisition_counts.get) if requisition_counts else None

        return {
            "total_alerts": len(alerts),
            "by_level": level_counts,
            "by_requisition_type": requisition_counts,
            "daily_average": round(daily_average, 2),
            "most_common_requisition": most_common_requisition,
            "last_30_days": len(recent_alerts)
        }

    def clear_old_alerts(self, days_to_keep: int = 90) -> int:
        """
        Limpia alertas antiguas

        Args:
            days_to_keep: Días de alertas a mantener

        Returns:
            Número de alertas eliminadas
        """
        if not os.path.exists(self.alerts_log_path):
            return 0

        try:
            with open(self.alerts_log_path, 'r', encoding='utf-8') as f:
                alerts = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return 0

        # Calcular fecha límite
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        # Filtrar alertas recientes
        initial_count = len(alerts)
        alerts = [
            alert for alert in alerts
            if datetime.fromisoformat(alert.get('logged_at', '').replace('Z', '+00:00')) >= cutoff_date
        ]

        # Guardar alertas filtradas
        with open(self.alerts_log_path, 'w', encoding='utf-8') as f:
            json.dump(alerts, f, indent=2, ensure_ascii=False)

        deleted_count = initial_count - len(alerts)
        self.logger.info(f"Limpieza de alertas completada. Eliminadas: {deleted_count}")

        return deleted_count

    def generate_alert_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Genera un reporte de alertas

        Args:
            days: Días a incluir en el reporte

        Returns:
            Reporte detallado
        """
        from datetime import datetime, timedelta

        # Obtener alertas del período
        start_date = datetime.now() - timedelta(days=days)
        alerts = self.get_alert_history(limit=1000)

        period_alerts = [
            alert for alert in alerts
            if datetime.fromisoformat(alert.get('logged_at', '').replace('Z', '+00:00')) >= start_date
        ]

        # Análisis por día
        daily_counts = {}
        for alert in period_alerts:
            date_str = alert.get('logged_at', '')[:10]  # YYYY-MM-DD
            daily_counts[date_str] = daily_counts.get(date_str, 0) + 1

        # Análisis por hora
        hourly_counts = {}
        for alert in period_alerts:
            hour = alert.get('logged_at', '')[11:13]  # HH
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

        # Top personas requisitoriadas
        person_counts = {}
        for alert in period_alerts:
            alert_info = alert.get('alert_info', {})
            person_id = alert_info.get('person_id')
            person_name = f"{alert_info.get('person_name', '')} {alert_info.get('person_lastname', '')}"

            if person_id:
                person_counts[f"{person_name} (ID: {person_id})"] = person_counts.get(
                    f"{person_name} (ID: {person_id})", 0) + 1

        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": datetime.now().isoformat(),
                "days": days
            },
            "summary": {
                "total_alerts": len(period_alerts),
                "daily_average": len(period_alerts) / days,
                "peak_day": max(daily_counts, key=daily_counts.get) if daily_counts else None,
                "peak_hour": max(hourly_counts, key=hourly_counts.get) if hourly_counts else None
            },
            "by_level": {
                level: len([a for a in period_alerts if a.get('alert_info', {}).get('alert_level') == level])
                for level in ["HIGH", "MEDIUM", "LOW"]
            },
            "daily_breakdown": daily_counts,
            "hourly_breakdown": hourly_counts,
            "top_requisitioned_persons": dict(sorted(person_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "generated_at": datetime.now().isoformat()
        }

    def test_alert_system(self) -> Dict[str, Any]:
        """
        Prueba el sistema de alertas con datos de ejemplo
        """
        # Crear alerta de prueba
        test_alert = AlertInfo(
            person_id=99999,
            person_name="Juan",
            person_lastname="Prueba",
            student_id="TEST001",
            requisition_type="Robo",
            confidence=85.5,
            detection_timestamp=datetime.now().isoformat(),
            image_path="/test/image.jpg",
            alert_level="HIGH",
            location="Campus Principal - Edificio A",
            additional_info={"test": True}
        )

        # Generar alerta
        result = self.generate_security_alert(test_alert)

        # Añadir información de prueba
        result["test_mode"] = True
        result["note"] = "Esta es una alerta de prueba del sistema"

        return result