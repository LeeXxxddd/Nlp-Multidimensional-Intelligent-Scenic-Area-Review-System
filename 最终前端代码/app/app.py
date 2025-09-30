from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_preprocessor import ReviewAnalyzer
from models.quality_evaluator import ReviewQualityEvaluator
from models.bert_classifier import BERTReviewClassifier
from models.similarity_detector import SimilarityDetector

# 设置模板路径和静态文件路径
template_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates')
static_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# 初始化模型
review_analyzer = ReviewAnalyzer()
quality_evaluator = ReviewQualityEvaluator()
similarity_detector = SimilarityDetector()

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传文件接口"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        
        # 保存文件
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        
        # 读取评论数据
        if file.filename.endswith('.csv'):
            reviews = review_analyzer.process_csv_file(file_path, '评论内容')
        else:
            reviews = review_analyzer.process_txt_file(file_path)
        
        return jsonify({
            'message': '文件上传成功',
            'filename': file.filename,
            'review_count': len(reviews),
            'file_path': file_path
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_reviews():
    """分析评论接口"""
    try:
        data = request.json
        file_path = data.get('file_path')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': '文件不存在'}), 400
        
        # 读取评论
        if file_path.endswith('.csv'):
            reviews = review_analyzer.process_csv_file(file_path, '评论内容')
        else:
            reviews = review_analyzer.process_txt_file(file_path)
        
        # 数据预处理
        unique_reviews, duplicate_reviews = review_analyzer.remove_duplicates(reviews)
        
        # 相似度检测
        similar_indices = review_analyzer.detect_similar_reviews(unique_reviews)
        filtered_reviews = [review for i, review in enumerate(unique_reviews) 
                          if i not in similar_indices]
        
        # 广告检测
        non_ad_reviews, ad_reviews = review_analyzer.detect_ads(filtered_reviews)
        
        # 情感分析
        positive, neutral, negative = review_analyzer.analyze_sentiment(non_ad_reviews)
        
        # 质量评估
        # ✅ 优化：限制参考集规模，提高处理效率
        quality_results = []
        reference_subset = non_ad_reviews[:1000]  # 只用前1000条作为参考集，速度大幅提升

        for i, review in enumerate(non_ad_reviews[:100]):
            print(f"[质量评估] 正在处理 {i+1}/100")  # 可在终端显示进度
            quality_result = quality_evaluator.evaluate_quality(review, reference_subset)
            quality_results.append({
                'review': review,
                'quality': quality_result
        })

        
        # 相似度聚类分析
        if len(non_ad_reviews) > 0:
            clusters = similarity_detector.detect_similar_clusters(non_ad_reviews[:50])
            copy_detection = similarity_detector.detect_copy_modification_network(non_ad_reviews[:50])
        else:
            clusters = []
            copy_detection = {'copy_modified_groups': [], 'similar_groups': []}
        
        # 统计结果
        analysis_result = {
            'total_reviews': len(reviews),
            'duplicate_count': len(duplicate_reviews),
            'similar_count': len(similar_indices),
            'ad_count': len(ad_reviews),
            'positive_count': len(positive),
            'neutral_count': len(neutral),
            'negative_count': len(negative),
            'quality_distribution': {
                '优秀': sum(1 for r in quality_results if r['quality']['quality_level'] == '优秀'),
                '良好': sum(1 for r in quality_results if r['quality']['quality_level'] == '良好'),
                '一般': sum(1 for r in quality_results if r['quality']['quality_level'] == '一般'),
                '较差': sum(1 for r in quality_results if r['quality']['quality_level'] == '较差')
            },
            'clusters': clusters,
            'copy_detection': copy_detection,
            'quality_results': quality_results[:20],  # 返回前20个详细结果
            'effective_reviews': len(positive) + len(negative),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(analysis_result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualize', methods=['POST'])
def create_visualizations():
    """生成可视化图表"""
    try:
        data = request.json
        analysis_result = data.get('analysis_result')
        
        if not analysis_result:
            return jsonify({'error': '没有分析结果'}), 400
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 评论类型分布饼图
        labels = ['正常评论', '重复评论', '相似评论', '广告评论']
        sizes = [
            analysis_result['effective_reviews'],
            analysis_result['duplicate_count'],
            analysis_result['similar_count'],
            analysis_result['ad_count']
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('评论类型分布')
        
        # 2. 情感分析结果柱状图
        sentiment_labels = ['正面', '中性', '负面']
        sentiment_counts = [
            analysis_result['positive_count'],
            analysis_result['neutral_count'],
            analysis_result['negative_count']
        ]
        
        axes[0, 1].bar(sentiment_labels, sentiment_counts, color=['green', 'gray', 'red'])
        axes[0, 1].set_title('情感分析结果')
        axes[0, 1].set_ylabel('评论数量')
        
        # 3. 质量分布柱状图
        quality_labels = list(analysis_result['quality_distribution'].keys())
        quality_counts = list(analysis_result['quality_distribution'].values())
        
        axes[1, 0].bar(quality_labels, quality_counts, color=['gold', 'silver', 'orange', 'red'])
        axes[1, 0].set_title('评论质量分布')
        axes[1, 0].set_ylabel('评论数量')
        
        # 4. 质量得分分布直方图
        if analysis_result.get('quality_results'):
            quality_scores = [r['quality']['total_score'] for r in analysis_result['quality_results']]
            axes[1, 1].hist(quality_scores, bins=20, color='skyblue', alpha=0.7)
            axes[1, 1].set_title('质量得分分布')
            axes[1, 1].set_xlabel('质量得分')
            axes[1, 1].set_ylabel('频次')
        
        plt.tight_layout()
        
        # 保存图表为base64编码
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return jsonify({
            'chart_image': f'data:image/png;base64,{img_base64}',
            'message': '可视化图表生成成功'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quality_evaluate', methods=['POST'])
def evaluate_single_review():
    """评估单条评论质量"""
    try:
        # ✅ 显式检查请求格式
        if not request.is_json:
            return jsonify({'error': '请求格式应为 JSON'}), 400

        data = request.get_json(force=True)
        review_text = data.get('review_text', '').strip()
        reference_texts = data.get('reference_texts', [])

        if not review_text:
            return jsonify({'error': '评论内容不能为空'}), 400

        print(f"[单条评估] 接收到评论: {review_text}")
        print(f"[参考数量] {len(reference_texts)} 条")

        # 质量评估
        quality_result = quality_evaluator.evaluate_quality(review_text, reference_texts)

        # 相似度检测（可选）
        similarity_result = None
        if reference_texts:
            all_texts = [review_text] + reference_texts
            similarity_matrix = similarity_detector.calculate_semantic_similarity(all_texts)
            similarity_result = {
                'max_similarity': float(similarity_matrix[0][1:].max()),
                'similarity_scores': similarity_matrix[0][1:].tolist()
            }
        detailed = quality_result.get('detailed_scores', {})
        response_payload = {
            'review_text': review_text,
            'quality_assessment': {
                'total_score': quality_result.get('total_score', 0),
                'quality_level': quality_result.get('quality_level', '一般'),
                'informativeness': detailed.get('information_score', 0),
                'subjectivity': detailed.get('subjectivity_score', 0.5),
                'readability': detailed.get('readability_score', 0),
                'detail_level': detailed.get('length_score', 0)
            },
            'problem_detection': {
                'is_duplicate': False,
                'is_similar': similarity_result is not None,
                'max_similarity': similarity_result['max_similarity'] if similarity_result else 0,
                'similarities': similarity_result['similarity_scores'] if similarity_result else [],
                'is_ad': False,
                'is_short': len(review_text) < 15
            },
            'recommendations': generate_recommendations(quality_result)
        }
        print("[单条评估返回]", json.dumps(response_payload, ensure_ascii=False, indent=2))
        return jsonify(response_payload)

        return jsonify({
            'review_text': review_text,
            'quality_assessment': {
                'total_score': quality_result.get('total_score', 0),
                'quality_level': quality_result.get('quality_level', '一般'),
                'informativeness': detailed.get('information_score', 0),
                'subjectivity': detailed.get('subjectivity_score', 0.5),  # ❗默认中性
                'readability': detailed.get('readability_score', 0),
                'detail_level': detailed.get('length_score', 0)
            },
            'problem_detection': {
                'is_duplicate': False,  # 若未实现可设为 False
                'is_similar': similarity_result is not None,
                'max_similarity': similarity_result['max_similarity'] if similarity_result else 0,
                'similarities': similarity_result['similarity_scores'] if similarity_result else [],
                'is_ad': False,  # 若未实现广告检测
                'is_short': len(review_text) < 15
            },
            'recommendations': generate_recommendations(quality_result)
        })


    except Exception as e:
        print(f"[质量评估错误] {e}")
        return jsonify({'error': str(e)}), 500


def generate_recommendations(quality_result):
    """生成改进建议"""
    recommendations = []
    scores = quality_result['detailed_scores']
    
    if scores['length_score'] < 0.5:
        recommendations.append('建议增加评论长度，提供更多详细信息')
    
    if scores['complexity_score'] < 0.5:
        recommendations.append('建议使用更丰富的词汇和句式结构')
    
    if scores['information_score'] < 0.5:
        recommendations.append('建议添加更多具体的体验细节，如服务、设施、价格等')
    
    if scores['originality_score'] < 0.5:
        recommendations.append('建议提供更原创的内容，避免模板化表达')
    
    if not recommendations:
        recommendations.append('评论质量良好，继续保持！')
    
    return recommendations

@app.route('/api/export_report', methods=['POST'])
def export_analysis_report():
    try:
        data = request.json
        analysis_result = data.get('analysis_result')

        if not analysis_result:
            return jsonify({'error': '没有分析结果'}), 400

        # 生成报告 HTML
        chart_base64 = generate_chart_base64(analysis_result)  # 新增这一行
        report_html = generate_analysis_report(analysis_result, chart_base64)


        # ✅ 构建绝对路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"analysis_report_{timestamp}.html"
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        report_dir = os.path.join(base_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, filename)

        # 保存 HTML 文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)

        # 下载
        return send_file(
            report_path,
            as_attachment=True,
            download_name=filename,
            mimetype='text/html'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_chart_base64(analysis_result):
    """根据分析结果生成 base64 图像字符串"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 评论类型分布
    labels = ['正常评论', '重复评论', '相似评论', '广告评论']
    sizes = [
        analysis_result['effective_reviews'],
        analysis_result['duplicate_count'],
        analysis_result['similar_count'],
        analysis_result['ad_count']
    ]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('评论类型分布')

    # 2. 情感分析结果柱状图
    sentiment_labels = ['正面', '中性', '负面']
    sentiment_counts = [
        analysis_result['positive_count'],
        analysis_result['neutral_count'],
        analysis_result['negative_count']
    ]
    axes[0, 1].bar(sentiment_labels, sentiment_counts, color=['green', 'gray', 'red'])
    axes[0, 1].set_title('情感分析结果')
    axes[0, 1].set_ylabel('评论数量')

    # 3. 评论质量分布
    quality_labels = list(analysis_result['quality_distribution'].keys())
    quality_counts = list(analysis_result['quality_distribution'].values())
    axes[1, 0].bar(quality_labels, quality_counts, color=['gold', 'silver', 'orange', 'red'])
    axes[1, 0].set_title('评论质量分布')
    axes[1, 0].set_ylabel('评论数量')

    # 4. 质量得分分布
    if analysis_result.get('quality_results'):
        quality_scores = [r['quality']['total_score'] for r in analysis_result['quality_results']]
        axes[1, 1].hist(quality_scores, bins=20, color='skyblue', alpha=0.7)
        axes[1, 1].set_title('质量得分分布')
        axes[1, 1].set_xlabel('质量得分')
        axes[1, 1].set_ylabel('频次')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()

    return f"data:image/png;base64,{img_base64}"


def generate_analysis_report(analysis_result, chart_base64=''):

    """生成HTML分析报告"""
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>景区评论质量分析报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; text-align: center; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; }}
            .quality-high {{ color: green; }}
            .quality-medium {{ color: orange; }}
            .quality-low {{ color: red; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>景区评论质量智能评估报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>数据概览</h2>
            <div class="metric">总评论数: {analysis_result['total_reviews']}</div>
            <div class="metric">有效评论数: {analysis_result['effective_reviews']}</div>
            <div class="metric">重复评论数: {analysis_result['duplicate_count']}</div>
            <div class="metric">相似评论数: {analysis_result['similar_count']}</div>
            <div class="metric">广告评论数: {analysis_result['ad_count']}</div>
        </div>
        
        <div class="section">
            <h2>情感分析结果</h2>
            <div class="metric">正面评论: {analysis_result['positive_count']}</div>
            <div class="metric">中性评论: {analysis_result['neutral_count']}</div>
            <div class="metric">负面评论: {analysis_result['negative_count']}</div>
        </div>
        
        <div class="section">
            <h2>质量评估结果</h2>
            <div class="metric quality-high">优秀: {analysis_result['quality_distribution']['优秀']}</div>
            <div class="metric quality-medium">良好: {analysis_result['quality_distribution']['良好']}</div>
            <div class="metric quality-medium">一般: {analysis_result['quality_distribution']['一般']}</div>
            <div class="metric quality-low">较差: {analysis_result['quality_distribution']['较差']}</div>
        </div>
        
        <div class="section">
            <h2>相似度聚类结果</h2>
            <p>检测到 {len(analysis_result['clusters'])} 个相似评论聚类</p>
            <p>发现 {len(analysis_result['copy_detection']['copy_modified_groups'])} 个疑似复制修改的评论组</p>
        </div>
        
        <div class="section">
            <h2>建议</h2>
            <ul>
                <li>建议过滤掉重复和相似度过高的评论</li>
                <li>关注广告评论的识别和处理</li>
                <li>优先展示高质量的评论内容</li>
                <li>建立评论质量监控机制</li>
            </ul>
        </div>
        <div class="section">
            <h2>可视化图表</h2>
            {"<img src='" + chart_base64 + "' alt='分析图表' style='width: 100%; max-width: 800px; border: 1px solid #ccc;' />" if chart_base64 else "<p>未生成图表</p>"}
        </div>
    </body>
    </html>
    """
    return html_template

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)