"""
报告生成模块

提供模型评估、监控和分析的报告生成功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path


class ModelReportGenerator:
    """
    模型报告生成器

    生成标准化的模型评估、监控和分析报告
    """

    def __init__(self, model_name: str = "model", output_dir: str = "./reports"):
        """
        初始化报告生成器

        Parameters:
        -----------
        model_name : str, default="model"
            模型名称
        output_dir : str, default="./reports"
            报告输出目录
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"ReportGenerator_{model_name}")

    def generate_evaluation_report(self,
                                 evaluation_results: Dict,
                                 feature_info: Optional[Dict] = None,
                                 model_info: Optional[Dict] = None) -> Dict:
        """
        生成模型评估报告

        Parameters:
        -----------
        evaluation_results : dict
            评估结果
        feature_info : dict, optional
            特征信息
        model_info : dict, optional
            模型信息

        Returns:
        --------
        report : dict
            评估报告
        """
        timestamp = datetime.now().isoformat()

        report = {
            'report_type': 'model_evaluation',
            'model_name': self.model_name,
            'generated_at': timestamp,
            'summary': self._create_evaluation_summary(evaluation_results),
            'detailed_metrics': evaluation_results,
            'feature_info': feature_info or {},
            'model_info': model_info or {},
            'recommendations': self._generate_evaluation_recommendations(evaluation_results)
        }

        return report

    def generate_monitoring_report(self,
                                 monitoring_results: List[Dict],
                                 time_range_hours: int = 24) -> Dict:
        """
        生成监控报告

        Parameters:
        -----------
        monitoring_results : list
            监控结果列表
        time_range_hours : int, default=24
            时间范围（小时）

        Returns:
        --------
        report : dict
            监控报告
        """
        timestamp = datetime.now().isoformat()
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        cutoff_str = cutoff_time.isoformat()

        # 过滤时间范围内的结果
        recent_results = [
            result for result in monitoring_results
            if result.get('timestamp', '') >= cutoff_str
        ]

        report = {
            'report_type': 'monitoring',
            'model_name': self.model_name,
            'generated_at': timestamp,
            'time_range_hours': time_range_hours,
            'monitoring_cycles': len(recent_results),
            'summary': self._create_monitoring_summary(recent_results),
            'alerts_analysis': self._analyze_alerts(recent_results),
            'performance_trends': self._analyze_performance_trends(recent_results),
            'recommendations': self._generate_monitoring_recommendations(recent_results)
        }

        return report

    def generate_drift_report(self,
                            drift_results: List[Dict],
                            feature_names: Optional[List[str]] = None) -> Dict:
        """
        生成特征漂移报告

        Parameters:
        -----------
        drift_results : list
            漂移检测结果列表
        feature_names : list, optional
            特征名称列表

        Returns:
        --------
        report : dict
            漂移报告
        """
        timestamp = datetime.now().isoformat()

        report = {
            'report_type': 'feature_drift',
            'model_name': self.model_name,
            'generated_at': timestamp,
            'total_features': len(feature_names) if feature_names else 0,
            'summary': self._create_drift_summary(drift_results),
            'drift_analysis': self._analyze_drift_patterns(drift_results),
            'feature_stability': self._analyze_feature_stability(drift_results, feature_names),
            'recommendations': self._generate_drift_recommendations(drift_results)
        }

        return report

    def generate_comprehensive_report(self,
                                    evaluation_results: Optional[Dict] = None,
                                    monitoring_results: Optional[List[Dict]] = None,
                                    drift_results: Optional[List[Dict]] = None,
                                    custom_sections: Optional[Dict] = None) -> Dict:
        """
        生成综合报告

        Parameters:
        -----------
        evaluation_results : dict, optional
            评估结果
        monitoring_results : list, optional
            监控结果
        drift_results : list, optional
            漂移结果
        custom_sections : dict, optional
            自定义章节

        Returns:
        --------
        report : dict
            综合报告
        """
        timestamp = datetime.now().isoformat()

        report = {
            'report_type': 'comprehensive',
            'model_name': self.model_name,
            'generated_at': timestamp,
            'executive_summary': {},
            'sections': {}
        }

        # 添加各个章节
        if evaluation_results:
            report['sections']['evaluation'] = self.generate_evaluation_report(evaluation_results)

        if monitoring_results:
            report['sections']['monitoring'] = self.generate_monitoring_report(monitoring_results)

        if drift_results:
            report['sections']['drift'] = self.generate_drift_report(drift_results)

        if custom_sections:
            report['sections'].update(custom_sections)

        # 生成执行摘要
        report['executive_summary'] = self._create_executive_summary(report['sections'])

        return report

    def export_report(self,
                     report: Dict,
                     filename: Optional[str] = None,
                     format: str = 'json') -> str:
        """
        导出报告

        Parameters:
        -----------
        report : dict
            报告内容
        filename : str, optional
            文件名
        format : str, default='json'
            导出格式（json, excel, html）

        Returns:
        --------
        filepath : str
            导出文件路径
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name}_{report['report_type']}_{timestamp}"

        if format.lower() == 'json':
            filepath = self.output_dir / f"{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        elif format.lower() == 'excel':
            filepath = self.output_dir / f"{filename}.xlsx"
            self._export_to_excel(report, filepath)

        elif format.lower() == 'html':
            filepath = self.output_dir / f"{filename}.html"
            self._export_to_html(report, filepath)

        else:
            raise ValueError(f"不支持的导出格式: {format}")

        self.logger.info(f"报告已导出到: {filepath}")
        return str(filepath)

    def _create_evaluation_summary(self, evaluation_results: Dict) -> Dict:
        """创建评估摘要"""
        summary = {
            'model_performance': 'unknown',
            'key_metrics': {},
            'strengths': [],
            'concerns': []
        }

        # 提取关键指标
        if 'basic_metrics' in evaluation_results:
            metrics = evaluation_results['basic_metrics']
            summary['key_metrics'] = {
                'auc': metrics.get('auc', 0),
                'ks': metrics.get('ks', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0)
            }

            # 判断模型性能
            auc = metrics.get('auc', 0)
            if auc >= 0.8:
                summary['model_performance'] = 'excellent'
                summary['strengths'].append('AUC表现优秀')
            elif auc >= 0.7:
                summary['model_performance'] = 'good'
                summary['strengths'].append('AUC表现良好')
            elif auc >= 0.6:
                summary['model_performance'] = 'fair'
                summary['concerns'].append('AUC表现一般')
            else:
                summary['model_performance'] = 'poor'
                summary['concerns'].append('AUC表现较差')

        return summary

    def _create_monitoring_summary(self, monitoring_results: List[Dict]) -> Dict:
        """创建监控摘要"""
        if not monitoring_results:
            return {'status': 'no_data', 'message': '无监控数据'}

        # 统计报警
        total_alerts = sum(len(result.get('alerts', [])) for result in monitoring_results)
        critical_alerts = sum(
            len([alert for alert in result.get('alerts', [])
                if alert.get('severity') in ['HIGH', 'CRITICAL']])
            for result in monitoring_results
        )

        # 分析趋势
        latest_result = monitoring_results[-1] if monitoring_results else {}
        overall_status = latest_result.get('overall_status', 'unknown')

        summary = {
            'overall_status': overall_status,
            'total_monitoring_cycles': len(monitoring_results),
            'total_alerts': total_alerts,
            'critical_alerts': critical_alerts,
            'alert_rate': total_alerts / len(monitoring_results) if monitoring_results else 0,
            'latest_metrics': latest_result.get('results', {})
        }

        return summary

    def _create_drift_summary(self, drift_results: List[Dict]) -> Dict:
        """创建漂移摘要"""
        if not drift_results:
            return {'status': 'no_data', 'message': '无漂移检测数据'}

        latest_result = drift_results[-1] if drift_results else {}

        # 统计漂移特征
        drift_features = latest_result.get('drifted_features', 0)
        total_features = latest_result.get('total_features', 0)
        drift_rate = latest_result.get('drift_rate', 0)

        summary = {
            'drift_status': latest_result.get('overall_status', 'unknown'),
            'total_features': total_features,
            'drifted_features': drift_features,
            'drift_rate': drift_rate,
            'stability_level': self._assess_stability_level(drift_rate)
        }

        return summary

    def _analyze_alerts(self, monitoring_results: List[Dict]) -> Dict:
        """分析报警模式"""
        all_alerts = []
        for result in monitoring_results:
            for alert in result.get('alerts', []):
                alert['timestamp'] = result.get('timestamp')
                all_alerts.append(alert)

        if not all_alerts:
            return {'total_alerts': 0, 'alert_types': {}, 'severity_distribution': {}}

        # 统计报警类型
        alert_types = {}
        severity_counts = {}

        for alert in all_alerts:
            alert_type = alert.get('type', 'unknown')
            severity = alert.get('severity', 'unknown')

            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'total_alerts': len(all_alerts),
            'alert_types': alert_types,
            'severity_distribution': severity_counts,
            'most_common_alert': max(alert_types.items(), key=lambda x: x[1])[0] if alert_types else None
        }

    def _analyze_performance_trends(self, monitoring_results: List[Dict]) -> Dict:
        """分析性能趋势"""
        metrics_over_time = []

        for result in monitoring_results:
            stability_metrics = result.get('results', {}).get('stability', {}).get('metrics', {})
            if stability_metrics:
                metrics_over_time.append({
                    'timestamp': result.get('timestamp'),
                    'auc': stability_metrics.get('current_auc'),
                    'ks': stability_metrics.get('current_ks'),
                    'auc_change': stability_metrics.get('auc_change'),
                    'ks_change': stability_metrics.get('ks_change')
                })

        if not metrics_over_time:
            return {'trend_status': 'no_data'}

        # 计算趋势
        recent_metrics = metrics_over_time[-5:]  # 最近5次
        auc_values = [m['auc'] for m in recent_metrics if m['auc'] is not None]

        trend_analysis = {
            'data_points': len(metrics_over_time),
            'auc_trend': self._calculate_trend(auc_values) if auc_values else 'stable',
            'latest_auc': auc_values[-1] if auc_values else None,
            'auc_volatility': np.std(auc_values) if len(auc_values) > 1 else 0
        }

        return trend_analysis

    def _analyze_drift_patterns(self, drift_results: List[Dict]) -> Dict:
        """分析漂移模式"""
        if not drift_results:
            return {'pattern_status': 'no_data'}

        # 收集所有漂移报警
        all_drift_alerts = []
        for result in drift_results:
            for alert in result.get('drift_alerts', []):
                alert['timestamp'] = result.get('timestamp')
                all_drift_alerts.append(alert)

        # 统计经常漂移的特征
        feature_drift_counts = {}
        for alert in all_drift_alerts:
            feature = alert.get('feature')
            if feature:
                feature_drift_counts[feature] = feature_drift_counts.get(feature, 0) + 1

        # 找出最不稳定的特征
        most_unstable = sorted(feature_drift_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'total_drift_events': len(all_drift_alerts),
            'unique_features_affected': len(feature_drift_counts),
            'most_unstable_features': most_unstable,
            'average_drift_events_per_cycle': len(all_drift_alerts) / len(drift_results) if drift_results else 0
        }

    def _analyze_feature_stability(self, drift_results: List[Dict], feature_names: Optional[List[str]]) -> Dict:
        """分析特征稳定性"""
        if not feature_names:
            return {'stability_status': 'no_feature_names'}

        # 为每个特征计算稳定性得分
        feature_stability = {}

        for feature in feature_names:
            drift_count = 0
            total_checks = len(drift_results)

            for result in drift_results:
                for alert in result.get('drift_alerts', []):
                    if alert.get('feature') == feature:
                        drift_count += 1
                        break

            stability_score = 1 - (drift_count / total_checks) if total_checks > 0 else 1
            feature_stability[feature] = {
                'stability_score': stability_score,
                'drift_frequency': drift_count / total_checks if total_checks > 0 else 0,
                'stability_level': self._assess_feature_stability(stability_score)
            }

        # 排序得到最稳定和最不稳定的特征
        sorted_features = sorted(feature_stability.items(), key=lambda x: x[1]['stability_score'], reverse=True)

        return {
            'feature_count': len(feature_names),
            'most_stable_features': sorted_features[:5],
            'least_stable_features': sorted_features[-5:],
            'average_stability': np.mean([info['stability_score'] for info in feature_stability.values()]),
            'stability_distribution': self._get_stability_distribution(feature_stability)
        }

    def _generate_evaluation_recommendations(self, evaluation_results: Dict) -> List[str]:
        """生成评估建议"""
        recommendations = []

        if 'basic_metrics' in evaluation_results:
            metrics = evaluation_results['basic_metrics']
            auc = metrics.get('auc', 0)
            ks = metrics.get('ks', 0)

            if auc < 0.7:
                recommendations.append("模型AUC较低，建议重新选择特征或调整模型参数")
            if ks < 0.2:
                recommendations.append("模型KS值较低，建议增强模型的区分能力")
            if auc > 0.9:
                recommendations.append("模型AUC很高，请检查是否存在数据泄露")

        return recommendations

    def _generate_monitoring_recommendations(self, monitoring_results: List[Dict]) -> List[str]:
        """生成监控建议"""
        recommendations = []

        # 分析报警频率
        alert_analysis = self._analyze_alerts(monitoring_results)
        total_alerts = alert_analysis['total_alerts']
        cycles = len(monitoring_results)

        if cycles > 0:
            alert_rate = total_alerts / cycles
            if alert_rate > 0.5:
                recommendations.append("报警频率较高，建议检查模型稳定性")
            if alert_rate > 0.8:
                recommendations.append("报警频率过高，建议立即检查数据质量和模型性能")

        return recommendations

    def _generate_drift_recommendations(self, drift_results: List[Dict]) -> List[str]:
        """生成漂移建议"""
        recommendations = []

        if not drift_results:
            return recommendations

        latest_result = drift_results[-1]
        drift_rate = latest_result.get('drift_rate', 0)

        if drift_rate > 0.3:
            recommendations.append("特征漂移率较高，建议重新训练模型")
        if drift_rate > 0.5:
            recommendations.append("特征漂移严重，建议立即停止使用当前模型")

        return recommendations

    def _create_executive_summary(self, sections: Dict) -> Dict:
        """创建执行摘要"""
        summary = {
            'overall_health': 'unknown',
            'key_findings': [],
            'critical_issues': [],
            'recommendations': []
        }

        # 汇总各部分的关键信息
        health_scores = []

        if 'evaluation' in sections:
            eval_summary = sections['evaluation']['summary']
            performance = eval_summary.get('model_performance', 'unknown')

            if performance in ['excellent', 'good']:
                health_scores.append(1)
                summary['key_findings'].append(f"模型评估表现{performance}")
            elif performance == 'fair':
                health_scores.append(0.5)
                summary['key_findings'].append("模型评估表现一般")
            else:
                health_scores.append(0)
                summary['critical_issues'].append("模型评估表现较差")

        if 'monitoring' in sections:
            monitor_summary = sections['monitoring']['summary']
            critical_alerts = monitor_summary.get('critical_alerts', 0)

            if critical_alerts == 0:
                health_scores.append(1)
                summary['key_findings'].append("监控状态正常")
            elif critical_alerts < 3:
                health_scores.append(0.7)
                summary['key_findings'].append(f"存在{critical_alerts}个高级报警")
            else:
                health_scores.append(0.3)
                summary['critical_issues'].append(f"存在{critical_alerts}个高级报警")

        if 'drift' in sections:
            drift_summary = sections['drift']['summary']
            drift_rate = drift_summary.get('drift_rate', 0)

            if drift_rate < 0.2:
                health_scores.append(1)
                summary['key_findings'].append("特征稳定性良好")
            elif drift_rate < 0.4:
                health_scores.append(0.6)
                summary['key_findings'].append(f"特征漂移率为{drift_rate:.1%}")
            else:
                health_scores.append(0.2)
                summary['critical_issues'].append(f"特征漂移严重({drift_rate:.1%})")

        # 计算总体健康度
        if health_scores:
            avg_health = np.mean(health_scores)
            if avg_health >= 0.8:
                summary['overall_health'] = 'healthy'
            elif avg_health >= 0.6:
                summary['overall_health'] = 'warning'
            else:
                summary['overall_health'] = 'critical'

        return summary

    def _export_to_excel(self, report: Dict, filepath: Path) -> None:
        """导出到Excel"""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 创建概览页
            overview_data = {
                'Item': ['Report Type', 'Model Name', 'Generated At'],
                'Value': [report.get('report_type'), report.get('model_name'), report.get('generated_at')]
            }
            pd.DataFrame(overview_data).to_excel(writer, sheet_name='Overview', index=False)

            # 根据报告类型添加详细页面
            if report.get('report_type') == 'comprehensive' and 'sections' in report:
                for section_name, section_data in report['sections'].items():
                    try:
                        # 将字典数据转换为DataFrame
                        section_df = pd.json_normalize(section_data)
                        section_df.to_excel(writer, sheet_name=section_name[:31], index=False)  # Excel工作表名称限制
                    except Exception as e:
                        self.logger.warning(f"无法导出章节 {section_name}: {e}")

    def _export_to_html(self, report: Dict, filepath: Path) -> None:
        """导出到HTML"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.get('model_name', 'Model')} Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .alert {{ color: red; font-weight: bold; }}
                .success {{ color: green; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report.get('model_name', 'Model')} Report</h1>
                <p>Report Type: {report.get('report_type', 'Unknown')}</p>
                <p>Generated: {report.get('generated_at', 'Unknown')}</p>
            </div>

            <div class="section">
                <h2>Report Content</h2>
                <pre>{json.dumps(report, indent=2, ensure_ascii=False, default=str)}</pre>
            </div>
        </body>
        </html>
        """

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return 'stable'

        # 简单线性趋势计算
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'

    def _assess_stability_level(self, drift_rate: float) -> str:
        """评估稳定性水平"""
        if drift_rate < 0.1:
            return 'very_stable'
        elif drift_rate < 0.2:
            return 'stable'
        elif drift_rate < 0.4:
            return 'moderately_stable'
        else:
            return 'unstable'

    def _assess_feature_stability(self, stability_score: float) -> str:
        """评估特征稳定性"""
        if stability_score >= 0.9:
            return 'very_stable'
        elif stability_score >= 0.7:
            return 'stable'
        elif stability_score >= 0.5:
            return 'moderately_stable'
        else:
            return 'unstable'

    def _get_stability_distribution(self, feature_stability: Dict) -> Dict:
        """获取稳定性分布"""
        distribution = {'very_stable': 0, 'stable': 0, 'moderately_stable': 0, 'unstable': 0}

        for feature_info in feature_stability.values():
            level = feature_info['stability_level']
            distribution[level] += 1

        return distribution


def generate_model_report(model_name: str,
                         evaluation_results: Optional[Dict] = None,
                         monitoring_results: Optional[List[Dict]] = None,
                         drift_results: Optional[List[Dict]] = None,
                         output_dir: str = "./reports",
                         export_format: str = 'json') -> str:
    """
    快捷函数：生成模型报告

    Parameters:
    -----------
    model_name : str
        模型名称
    evaluation_results : dict, optional
        评估结果
    monitoring_results : list, optional
        监控结果
    drift_results : list, optional
        漂移结果
    output_dir : str, default="./reports"
        输出目录
    export_format : str, default='json'
        导出格式

    Returns:
    --------
    filepath : str
        报告文件路径
    """
    generator = ModelReportGenerator(model_name, output_dir)

    report = generator.generate_comprehensive_report(
        evaluation_results=evaluation_results,
        monitoring_results=monitoring_results,
        drift_results=drift_results
    )

    return generator.export_report(report, format=export_format)