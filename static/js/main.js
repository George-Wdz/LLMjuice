/**
 * LLMjuice Web Application JavaScript
 * 主要的客户端交互逻辑
 */

// 全局变量
let isProcessing = false;
let statusCheckInterval = null;

// 页面加载完成后初始化
$(document).ready(function() {
    initializeApp();
});

/**
 * 初始化应用程序
 */
function initializeApp() {
    setupFileUpload();
    setupDragAndDrop();
    refreshFiles();
    updateStats();

    // 设置定时更新
    setInterval(updateStats, 30000); // 每30秒更新统计信息

    console.log('LLMjuice Web Application initialized');
}

/**
 * 设置文件上传功能
 */
function setupFileUpload() {
    $('#file-input').change(function(e) {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            uploadFiles(files);
        }
    });
}

/**
 * 设置拖拽上传功能
 */
function setupDragAndDrop() {
    const uploadArea = $('#upload-area');

    uploadArea.on('dragover', function(e) {
        e.preventDefault();
        uploadArea.addClass('dragover');
    });

    uploadArea.on('dragleave', function(e) {
        e.preventDefault();
        uploadArea.removeClass('dragover');
    });

    uploadArea.on('drop', function(e) {
        e.preventDefault();
        uploadArea.removeClass('dragover');

        const files = Array.from(e.originalEvent.dataTransfer.files).filter(file =>
            file.name.toLowerCase().endsWith('.pdf')
        );

        if (files.length > 0) {
            uploadFiles(files);
        }
    });

    uploadArea.on('click', function() {
        $('#file-input').click();
    });
}

/**
 * 上传文件
 */
function uploadFiles(files) {
    if (files.length === 0) return;

    const formData = new FormData();
    files.forEach(file => {
        formData.append('files', file);
    });

    // 显示上传进度
    showUploadProgress();

    $.ajax({
        url: '/upload',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            hideUploadProgress();
            if (response.success) {
                showAlert('success', `成功上传 ${response.files.length} 个文件`);
                refreshFiles();
                updateStats();
            } else {
                showAlert('error', response.error);
            }
        },
        error: function(xhr) {
            hideUploadProgress();
            const error = xhr.responseJSON ? xhr.responseJSON.error : '上传失败';
            showAlert('error', error);
        }
    });
}

/**
 * 显示上传进度
 */
function showUploadProgress() {
    $('#upload-progress').removeClass('d-none');
    $('#upload-progress .progress-bar').css('width', '100%')
        .addClass('progress-bar-striped progress-bar-animated');
}

/**
 * 隐藏上传进度
 */
function hideUploadProgress() {
    $('#upload-progress').addClass('d-none');
    $('#upload-progress .progress-bar').css('width', '0%')
        .removeClass('progress-bar-striped progress-bar-animated');
}

/**
 * 刷新文件列表
 */
function refreshFiles() {
    $.get('/files', function(response) {
        displayFileList(response.files);
        updateFileCount(response.files.length);
    }).fail(function() {
        showAlert('error', '获取文件列表失败');
    });
}

/**
 * 显示文件列表
 */
function displayFileList(files) {
    const fileList = $('#file-list');

    if (files.length === 0) {
        fileList.html('<p class="text-muted text-center">暂无PDF文件</p>');
        return;
    }

    let html = '';
    files.forEach(function(file) {
        html += `
            <div class="file-item">
                <div class="file-info">
                    <div class="file-name">
                        <i class="fas fa-file-pdf text-danger me-2"></i>${file.name}
                    </div>
                    <div class="file-meta">
                        <span class="me-3"><i class="fas fa-hdd me-1"></i>${file.size}</span>
                        <span><i class="fas fa-clock me-1"></i>${file.modified}</span>
                    </div>
                </div>
                <div class="file-actions">
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteFile('${file.relative_path}')" title="删除文件">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
    });

    fileList.html(html);
}

/**
 * 更新文件计数
 */
function updateFileCount(count) {
    $('#pdf-count').text(count);
}

/**
 * 删除文件
 */
function deleteFile(filename) {
    if (!confirm('确定要删除这个文件吗？')) {
        return;
    }

    $.ajax({
        url: `/delete_file/${filename}`,
        type: 'POST',
        success: function(response) {
            if (response.success) {
                showAlert('success', response.message);
                refreshFiles();
                updateStats();
            } else {
                showAlert('error', response.error);
            }
        },
        error: function(xhr) {
            const error = xhr.responseJSON ? xhr.responseJSON.error : '删除失败';
            showAlert('error', error);
        }
    });
}

/**
 * 开始处理
 */
function startProcessing() {
    if (isProcessing) {
        showAlert('warning', '正在处理中，请等待当前处理完成');
        return;
    }

    // 重置状态标记
    resetProcessingState();

    // 重置进度条
    $('#status-progress').css('width', '0%').addClass('progress-bar-striped progress-bar-animated');
    $('#overall-progress-bar').css('width', '0%').addClass('progress-bar-striped progress-bar-animated');
    $('#overall-progress-text').text('0%');

    // 重置步骤状态
    $('.step-card').removeClass('active completed');
    $('.step-icon i').removeClass('fa-check fa-spinner fa-spin fa-check-circle')
                   .addClass('fa-play-circle');
    $('.step-status').remove();

    const startBtn = $('#start-process-btn');
    const originalText = startBtn.html();
    startBtn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin me-2"></i>正在启动处理...');

    $.ajax({
        url: '/process',
        type: 'POST',
        success: function(response) {
            if (response.success) {
                showAlert('success', '处理已启动，请等待完成');
                startStatusMonitoring();
                // 显示处理中状态
                $('#results-section').html(`
                    <div class="text-center text-muted py-4">
                        <i class="fas fa-cogs fa-3x mb-3 text-primary"></i>
                        <p class="mb-0">正在处理中，请稍候...</p>
                    </div>
                `);
            } else {
                showAlert('error', response.error);
                startBtn.prop('disabled', false).html(originalText);
            }
        },
        error: function(xhr) {
            const error = xhr.responseJSON ? xhr.responseJSON.error : '启动处理失败';
            showAlert('error', error);
            startBtn.prop('disabled', false).html(originalText);
        }
    });
}

/**
 * 开始状态监控
 */
function startStatusMonitoring() {
    isProcessing = true;
    $('#processing-status').removeClass('d-none');

    // 立即检查一次状态
    checkProcessingStatus();

    // 每2秒检查一次状态
    statusCheckInterval = setInterval(checkProcessingStatus, 2000);
}

/**
 * 停止状态监控
 */
function stopStatusMonitoring() {
    isProcessing = false;
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

/**
 * 检查处理状态
 */
function checkProcessingStatus() {
    $.get('/status', function(status) {
        // 记录上一个状态
        const wasProcessing = lastProcessingState ? lastProcessingState.is_processing : true;

        updateProcessingDisplay(status);

        if (!status.is_processing && wasProcessing) {
            // 处理刚刚完成
            stopStatusMonitoring();
            const startBtn = $('#start-process-btn');
            startBtn.prop('disabled', false).html('<i class="fas fa-rocket me-2"></i>开始处理');

            if (status.error) {
                showAlert('error', '处理失败: ' + status.error);
            } else {
                // 只在真正完成时显示消息
                if (status.files && status.files.results.train_final) {
                    showAlert('success', '🎉 处理完成！训练数据已生成');
                    updateResults();
                } else {
                    showAlert('warning', '处理已结束，但没有生成新的训练数据');
                }
            }
        }

        // 更新状态记录
        lastProcessingState = status;
    }).fail(function() {
        console.error('Failed to check processing status');
    });
}

/**
 * 更新处理显示
 */
function updateProcessingDisplay(status) {
    // 更新详细进度条
    $('#status-progress').css('width', status.progress + '%');

    // 更新整体进度条
    $('#overall-progress-bar').css('width', status.progress + '%');
    $('#overall-progress-text').text(Math.round(status.progress) + '%');

    // 完成时停止动画
    if (status.progress >= 100) {
        $('#overall-progress-bar').removeClass('progress-bar-striped progress-bar-animated');
    }

    // 更新消息
    $('#status-message').text(status.message);

    // 更新计时器
    if (status.elapsed_time) {
        $('#status-timer').text(`已用时: ${status.elapsed_time}`);
    }

    // 更新步骤状态
    updateStepStatus(status.current_step);
}

/**
 * 更新步骤状态 - 增强版
 */
function updateStepStatus(currentStep) {
    $('.step-card').removeClass('active completed');

    const steps = ['upload', 'ocr', 'split', 'generate', 'complete'];
    const currentIndex = steps.indexOf(currentStep);

    $('.step-card').each(function() {
        const stepId = $(this).data('step');
        const stepIndex = steps.indexOf(stepId);

        const stepCard = $(this);
        const stepIcon = stepCard.find('.step-icon i');
        const stepTitle = stepCard.find('.card-title');

        // 重置图标
        stepIcon.removeClass('fa-check fa-spinner fa-spin fa-play-circle fa-eye fa-cut fa-comments fa-check-circle');
        stepCard.find('.step-status').remove();

        if (stepIndex < currentIndex) {
            // 已完成步骤
            stepCard.addClass('completed');
            stepIcon.addClass('fa-check-circle');
            stepTitle.after('<span class="badge bg-success ms-2 step-status">已完成</span>');
        } else if (stepIndex === currentIndex) {
            // 当前步骤
            stepCard.addClass('active');
            stepIcon.addClass('fa-spinner fa-spin');
            stepTitle.after('<span class="badge bg-primary ms-2 step-status">进行中</span>');
        } else {
            // 未开始步骤
            stepIcon.addClass('fa-play-circle');
        }
    });
}

/**
 * 设置步骤完成状态
 */
function setStepCompleted(stepId) {
    const stepCard = $(`.step-card[data-step="${stepId}"]`);
    stepCard.removeClass('active').addClass('completed');
    stepCard.find('.step-icon i').removeClass('fa-spinner fa-spin fa-play-circle')
                                   .addClass('fa-check-circle');
    stepCard.find('.step-status').removeClass('bg-primary').addClass('bg-success').text('已完成');
}

/**
 * 更新统计信息
 */
function updateStats() {
    $.get('/status', function(status) {
        if (status.files) {
            $('#pdf-count').text(status.files.pdf_count);

            let resultCount = 0;
            if (status.files.results.train_final) {
                resultCount += 1;
            }
            resultCount += status.files.results.markdown_files.length;
            resultCount += status.files.results.split_files.length;
            resultCount += status.files.results.train_files.length;

            $('#result-count').text(resultCount);

            // 更新最后更新时间
            const now = new Date().toLocaleString('zh-CN');
            $('#last-update').text(`最后更新: ${now}`);
        }
    });
}

// 全局变量跟踪处理状态
let hasShownCompletionMessage = false;
let lastProcessingState = null;

/**
 * 更新结果区域 - 修复重复弹出问题
 */
function updateResults() {
    $.get('/status', function(status) {
        // 检查是否刚刚完成处理
        const justCompleted = status.files && status.files.results.train_final &&
                           !status.is_processing &&
                           lastProcessingState && lastProcessingState.is_processing;

        if (justCompleted && !hasShownCompletionMessage) {
            const result = status.files.results.train_final;
            const resultsHtml = `
                <div class="alert alert-success fade-in">
                    <h6><i class="fas fa-check-circle me-1"></i>处理完成!</h6>
                    <p class="mb-2">最终训练数据文件已生成:</p>
                    <a href="${result.download_url}" class="btn btn-success btn-sm">
                        <i class="fas fa-download me-1"></i>下载 ${result.name}
                    </a>
                    <div class="mt-2 small text-muted">
                        大小: ${result.size} | 生成时间: ${result.modified}
                    </div>
                </div>
            `;
            $('#results-section').html(resultsHtml);
            hasShownCompletionMessage = true;
        }

        // 更新状态记录
        lastProcessingState = status;
    });
}

/**
 * 重置处理状态标记
 */
function resetProcessingState() {
    hasShownCompletionMessage = false;
    lastProcessingState = null;
}

/**
 * 清空结果
 */
function clearResults() {
    if (!confirm('确定要清空所有处理结果吗？这将删除所有生成的文件。')) {
        return;
    }

    // 这里可以添加清空结果的逻辑
    // 比如调用后端API删除结果文件
    showAlert('info', '清空结果功能待实现');
}

/**
 * 测试MinerU配置
 */
function testMinerUConfig() {
    const apiKey = $('#MinerU_KEY').val().trim();
    if (!apiKey) {
        showAlert('error', '请先输入MinerU API密钥');
        return;
    }

    const $testBtn = $('button[onclick="testMinerUConfig()"]');
    const originalText = $testBtn.html();
    $testBtn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin me-1"></i>测试中...');

    // 模拟测试 - 实际应用中这里应该调用真实的API测试
    setTimeout(function() {
        $testBtn.prop('disabled', false).html(originalText);

        // 简单的API密钥格式验证
        if (apiKey.startsWith('ey') && apiKey.length > 100) {
            showAlert('success', 'MinerU API密钥格式正确');
            $('#test-results').html('<div class="alert alert-success">✅ MinerU配置验证通过</div>');
        } else {
            showAlert('error', 'MinerU API密钥格式不正确');
            $('#test-results').html('<div class="alert alert-danger">❌ MinerU配置验证失败</div>');
        }
    }, 2000);
}

/**
 * 测试AI配置
 */
function testAIConfig() {
    const apiKey = $('#API_KEY').val().trim();
    const baseUrl = $('#BASE_URL').val().trim();
    const modelName = $('#MODEL_NAME').val() === 'custom' ?
        $('#custom-model-name').val().trim() : $('#MODEL_NAME').val();

    if (!apiKey || !baseUrl || !modelName) {
        showAlert('error', '请先完整填写AI配置信息');
        return;
    }

    const $testBtn = $('button[onclick="testAIConfig()"]');
    const originalText = $testBtn.html();
    $testBtn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin me-1"></i>测试中...');

    // 模拟测试 - 实际应用中这里应该调用真实的API测试
    setTimeout(function() {
        $testBtn.prop('disabled', false).html(originalText);

        // 简单的配置格式验证
        if (apiKey.startsWith('sk-') && baseUrl.startsWith('http')) {
            showAlert('success', 'AI配置格式正确');
            $('#test-results').html('<div class="alert alert-success">✅ AI配置验证通过</div>');
        } else {
            showAlert('error', 'AI配置格式不正确');
            $('#test-results').html('<div class="alert alert-danger">❌ AI配置验证失败</div>');
        }
    }, 2000);
}

/**
 * 显示提示消息
 */
function showAlert(type, message) {
    const alertClass = type === 'error' ? 'danger' : type;
    const icon = type === 'success' ? 'check-circle' :
                 type === 'error' ? 'exclamation-circle' :
                 type === 'warning' ? 'exclamation-triangle' : 'info-circle';

    const alertHtml = `
        <div class="alert alert-${alertClass} alert-dismissible fade show" role="alert">
            <i class="fas fa-${icon} me-1"></i>${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    // 在页面顶部显示提示
    $('main.container').prepend(alertHtml);

    // 自动隐藏提示消息
    setTimeout(function() {
        $('.alert').fadeOut();
    }, 5000);
}

/**
 * 格式化文件大小
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * 格式化时间
 */
function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}小时${minutes}分${secs}秒`;
    } else if (minutes > 0) {
        return `${minutes}分${secs}秒`;
    } else {
        return `${secs}秒`;
    }
}

// 错误处理
window.addEventListener('error', function(e) {
    console.error('JavaScript Error:', e.error);
});

window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled Promise Rejection:', e.reason);
});