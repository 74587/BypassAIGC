import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { 
  FileText, History, LogOut, Play, 
  Users, Clock, AlertCircle, CheckCircle, Trash2, Info 
} from 'lucide-react';
import { optimizationAPI } from '../api';

const WorkspacePage = () => {
  const [text, setText] = useState('');
  const [processingMode, setProcessingMode] = useState('paper_polish_enhance');
  const [sessions, setSessions] = useState([]);
  const [queueStatus, setQueueStatus] = useState(null);
  const [activeSession, setActiveSession] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    loadSessions();
    loadQueueStatus();
    
    // 降低队列状态更新频率到10秒，减少服务器负载
    const interval = setInterval(loadQueueStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // 如果有活跃会话,每3秒更新进度（从2秒增加到3秒，减少请求频率）
    if (activeSession) {
      const interval = setInterval(() => {
        updateSessionProgress(activeSession);
      }, 3000);
      return () => clearInterval(interval);
    }
  }, [activeSession]);

  const loadSessions = async () => {
    try {
      setIsLoadingSessions(true);
      const response = await optimizationAPI.listSessions();
      setSessions(response.data);
      
      // 查找正在处理的会话
      const processing = response.data.find(
        s => s.status === 'processing' || s.status === 'queued'
      );
      if (processing) {
        setActiveSession(processing.session_id);
      }
    } catch (error) {
      console.error('加载会话失败:', error);
    } finally {
      setIsLoadingSessions(false);
    }
  };

  const loadQueueStatus = async () => {
    try {
      const response = await optimizationAPI.getQueueStatus(activeSession);
      setQueueStatus(response.data);
    } catch (error) {
      console.error('加载队列状态失败:', error);
    }
  };

  const updateSessionProgress = async (sessionId) => {
    try {
      const response = await optimizationAPI.getSessionProgress(sessionId);
      const progress = response.data;
      
      // 更新会话列表中的进度
      setSessions(prev =>
        prev.map(s =>
          s.session_id === sessionId
            ? { ...s, ...progress }
            : s
        )
      );
      
      // 如果会话完成,刷新列表
      if (progress.status === 'completed' || progress.status === 'failed') {
        setActiveSession(null);
        loadSessions();
        
        if (progress.status === 'completed') {
          toast.success('优化完成!');
        } else {
          toast.error(`优化失败: ${progress.error_message}`);
        }
      }
    } catch (error) {
      console.error('更新进度失败:', error);
    }
  };

  const handleStartOptimization = async () => {
    if (!text.trim()) {
      toast.error('请输入要优化的文本');
      return;
    }

    if (isSubmitting) {
      return;
    }

    try {
      setIsSubmitting(true);
      const response = await optimizationAPI.startOptimization({
        original_text: text,
        processing_mode: processingMode,
      });
      
      setActiveSession(response.data.session_id);
      toast.success('优化任务已启动');
      setText('');
      loadSessions();
    } catch (error) {
      toast.error('启动优化失败: ' + error.response?.data?.detail);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('cardKey');
    navigate('/');
  };

  const handleDeleteSession = async (event, session) => {
    event.stopPropagation();

    const confirmDelete = window.confirm('确认删除该会话及其结果吗?');
    if (!confirmDelete) {
      return;
    }

    try {
      await optimizationAPI.deleteSession(session.session_id);
      if (activeSession === session.session_id) {
        setActiveSession(null);
      }
      toast.success('会话已删除');
      await loadSessions();
    } catch (error) {
      console.error('删除会话失败:', error);
      toast.error(error.response?.data?.detail || '删除会话失败');
    }
  };

  const handleViewSession = (sessionId) => {
    navigate(`/session/${sessionId}`);
  };

  const handleRetrySegment = async (event, session) => {
    event.stopPropagation();

    if (session.status !== 'failed') {
      return;
    }

    const confirmRetry = window.confirm('检测到会话执行失败。是否继续处理未完成的段落?');
    if (!confirmRetry) {
      return;
    }

    try {
      const response = await optimizationAPI.retryFailedSegments(session.session_id);
      setActiveSession(session.session_id);
      toast.success(response.data?.message || '已重新继续处理未完成段落');
      await loadSessions();
    } catch (error) {
      console.error('重试失败:', error);
      toast.error(error.response?.data?.detail || '重试失败，请稍后再试');
    }
  };

  const renderStatusAction = (session) => {
    if (session.status === 'failed') {
      return (
        <button
          onClick={(event) => handleRetrySegment(event, session)}
          className="px-2 py-1 text-xs bg-yellow-100 text-yellow-700 rounded hover:bg-yellow-200"
        >
          继续处理
        </button>
      );
    }
    return null;
  };


  return (
    <div className="min-h-screen bg-gray-50">
      {/* 顶部导航栏 */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <FileText className="w-8 h-8 text-blue-600" />
              <h1 className="text-xl font-bold text-gray-800">
                AI 论文润色增强系统
              </h1>
            </div>
            
            <div className="flex items-center gap-4">
              {/* 队列状态 */}
              {queueStatus && (
                <div className="flex items-center gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <Users className="w-4 h-4 text-gray-500" />
                    <span className="text-gray-600">
                      {queueStatus.current_users}/{queueStatus.max_users} 使用中
                    </span>
                  </div>
                  {queueStatus.queue_length > 0 && (
                    <div className="flex items-center gap-2">
                      <Clock className="w-4 h-4 text-orange-500" />
                      <span className="text-orange-600">
                        {queueStatus.queue_length} 人排队
                      </span>
                    </div>
                  )}
                </div>
              )}
              
              <button
                onClick={handleLogout}
                className="flex items-center gap-2 px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
              >
                <LogOut className="w-5 h-5" />
                退出
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 左侧 - 输入区域 */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-blue-50 border border-blue-200 text-blue-700 text-sm rounded-lg p-4 flex items-start gap-3">
              <Info className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-medium mb-1">处理模式说明</p>
                <p>
                  {processingMode === 'paper_polish' && '仅进行论文润色，提升文本的学术性和表达质量。'}
                  {processingMode === 'paper_polish_enhance' && '先进行论文润色，然后自动进行原创性增强，两阶段处理。'}
                  {processingMode === 'emotion_polish' && '专为感情文章设计，生成更自然、更具人性化的表达。'}
                </p>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-sm p-6">
              <h2 className="text-lg font-semibold text-gray-800 mb-4">
                输入文本内容
              </h2>
              
              {/* 处理模式选择 */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  选择处理模式
                </label>
                <div className="space-y-2">
                  <label className="flex items-center p-3 border border-gray-300 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
                    <input
                      type="radio"
                      name="processingMode"
                      value="paper_polish"
                      checked={processingMode === 'paper_polish'}
                      onChange={(e) => setProcessingMode(e.target.value)}
                      className="mr-3"
                    />
                    <div>
                      <div className="font-medium text-gray-800">论文润色</div>
                      <div className="text-sm text-gray-600">仅进行论文润色，提升学术表达质量</div>
                    </div>
                  </label>
                  
                  <label className="flex items-center p-3 border border-gray-300 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
                    <input
                      type="radio"
                      name="processingMode"
                      value="paper_polish_enhance"
                      checked={processingMode === 'paper_polish_enhance'}
                      onChange={(e) => setProcessingMode(e.target.value)}
                      className="mr-3"
                    />
                    <div>
                      <div className="font-medium text-gray-800">论文润色 + 论文增强</div>
                      <div className="text-sm text-gray-600">先润色后增强，提升原创性和学术水平</div>
                    </div>
                  </label>
                  
                  <label className="flex items-center p-3 border border-gray-300 rounded-lg cursor-pointer hover:bg-gray-50 transition-colors">
                    <input
                      type="radio"
                      name="processingMode"
                      value="emotion_polish"
                      checked={processingMode === 'emotion_polish'}
                      onChange={(e) => setProcessingMode(e.target.value)}
                      className="mr-3"
                    />
                    <div>
                      <div className="font-medium text-gray-800">感情文章润色</div>
                      <div className="text-sm text-gray-600">适合今日头条等平台，生成更自然的表达</div>
                    </div>
                  </label>
                </div>
              </div>
              
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="在此粘贴您的内容..."
                className="w-full h-64 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              />
              
              <div className="mt-4 flex items-center justify-between">
                <div className="text-sm text-gray-600">
                  字数: {text.length}
                </div>
                
                <button
                  onClick={handleStartOptimization}
                  disabled={!text.trim() || activeSession || isSubmitting}
                  className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-semibold py-2 px-6 rounded-lg transition-colors"
                >
                  {isSubmitting ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      提交中...
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      开始优化
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* 活跃会话进度 */}
            {activeSession && sessions.find(s => s.session_id === activeSession) && (
              <div className="bg-white rounded-xl shadow-sm p-6">
                <h2 className="text-lg font-semibold text-gray-800 mb-4">
                  优化进度
                </h2>
                
                {(() => {
                  const session = sessions.find(s => s.session_id === activeSession);
                  const getStageName = (stage) => {
                    if (stage === 'polish') return '论文润色';
                    if (stage === 'emotion_polish') return '感情文章润色';
                    if (stage === 'enhance') return '原创性增强';
                    return stage;
                  };
                  return (
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <span className="text-gray-600">
                            阶段: {getStageName(session.current_stage)}
                          </span>
                          <span className="font-medium text-blue-600">
                            {session.progress.toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-3">
                          <div
                            className="bg-blue-600 h-3 rounded-full transition-all duration-300"
                            style={{ width: `${session.progress}%` }}
                          />
                        </div>
                      </div>
                      
                      <div className="text-sm text-gray-600">
                        处理中: 第 {session.current_position + 1} / {session.total_segments} 段
                      </div>
                      
                      {session.status === 'queued' && queueStatus?.your_position && (
                        <div className="flex items-center gap-2 text-sm text-orange-600">
                          <Clock className="w-4 h-4" />
                          <span>
                            队列位置: {queueStatus.your_position} 
                            (预计等待 {Math.ceil(queueStatus.estimated_wait_time / 60)} 分钟)
                          </span>
                        </div>
                      )}
                    </div>
                  );
                })()}
              </div>
            )}
          </div>

          {/* 右侧 - 历史会话 */}
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-sm p-6">
              <div className="flex items-center gap-2 mb-4">
                <History className="w-5 h-5 text-gray-600" />
                <h2 className="text-lg font-semibold text-gray-800">
                  历史会话
                </h2>
              </div>
              
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {isLoadingSessions ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                  </div>
                ) : sessions.length === 0 ? (
                  <p className="text-gray-500 text-sm text-center py-8">
                    暂无会话记录
                  </p>
                ) : (
                  sessions.map((session) => (
                    <div
                      key={session.id}
                      onClick={() => handleViewSession(session.session_id)}
                      className="p-4 border border-gray-200 rounded-lg hover:border-blue-500 hover:shadow-md transition-all cursor-pointer"
                    >
                      <div className="flex items-start justify-between mb-2 gap-2">
                        <div className="flex items-center gap-2">
                          {session.status === 'completed' && (
                            <CheckCircle className="w-5 h-5 text-green-500" />
                          )}
                          {session.status === 'processing' && (
                            <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                          )}
                          {session.status === 'failed' && (
                            <AlertCircle className="w-5 h-5 text-red-500" />
                          )}
                          <span className="text-sm font-medium text-gray-700">
                            {session.status === 'completed' && '已完成'}
                            {session.status === 'processing' && '处理中'}
                            {session.status === 'queued' && '排队中'}
                            {session.status === 'failed' && '失败'}
                          </span>
                          {renderStatusAction(session)}
                        </div>
                        
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-gray-500">
                            {new Date(session.created_at).toLocaleDateString()}
                          </span>
                          <button
                            onClick={(event) => handleDeleteSession(event, session)}
                            className="p-1 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded" 
                            title="删除会话"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                      
                      <p className="text-sm text-gray-600 line-clamp-2">
                        {session.original_text?.substring(0, 100)}...
                      </p>
                      
                      {session.status === 'processing' && (
                        <div className="mt-2">
                          <div className="w-full bg-gray-200 rounded-full h-1.5">
                            <div
                              className="bg-blue-600 h-1.5 rounded-full"
                              style={{ width: `${session.progress}%` }}
                            />
                          </div>
                        </div>
                      )}
                      {session.status === 'failed' && session.current_position < session.total_segments && (
                        <p className="text-xs text-red-500 mt-2">
                          失败原因: {session.error_message || '网络超时，建议点击“继续处理”重试剩余段落。'}
                        </p>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WorkspacePage;
