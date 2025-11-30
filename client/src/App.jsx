import React, { useState, useEffect } from 'react';
import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Activity, Brain, Settings, TrendingUp, TrendingDown, RefreshCw, Clock, AlertCircle, HelpCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

// --- СТИЛІ (Мінімалістичний дизайн + Глобальні стилі скролбару) ---
const styles = {
  container: "min-h-screen bg-[#0f172a] text-slate-200 font-sans p-4 md:p-8",
  header: "max-w-6xl mx-auto mb-8 flex flex-col md:flex-row justify-between items-center gap-4",
  title: "text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent",
  grid: "max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6",
  // Чистий, темний стиль карток без зайвих ефектів
  card: "bg-[#1e293b]/80 backdrop-blur-md border border-slate-700/50 rounded-2xl p-6 shadow-xl relative overflow-hidden",
  input: "w-full bg-[#0f172a] border border-slate-700 rounded-lg px-4 py-2 text-slate-200 focus:outline-none focus:border-cyan-500 transition-colors",
  button: "w-full py-3 rounded-lg font-semibold transition-all duration-200 flex items-center justify-center gap-2",
  label: "block text-sm font-medium text-slate-400 mb-1",
  statLabel: "text-sm text-slate-400 uppercase tracking-wider font-medium flex items-center gap-1",
  statValue: "text-2xl font-bold text-white mt-1",
};

const API_URL = "http://127.0.0.1:5000";

export default function App() {
  const [status, setStatus] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [isServerOnline, setIsServerOnline] = useState(true);
  
  const [days, setDays] = useState(365);
  const [epochs, setEpochs] = useState(20);
  const [isSaving, setIsSaving] = useState(false);
  const [isDirty, setIsDirty] = useState(false);

  // 1. Отримання даних з сервера
  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/status`);
      if (!res.ok) throw new Error("Server offline");
      const data = await res.json();
      setStatus(data);
      setIsServerOnline(true);
      
      // Оновлюємо графік
      if (data.result?.chart_data?.length > 0) {
          setChartData(data.result.chart_data);
      }
      
      // Синхронізуємо налаштування (якщо користувач не редагує їх зараз)
      if (!isSaving && !isDirty) {
        setDays(data.config.days_data);
        setEpochs(data.config.epochs);
      }

    } catch (err) {
      console.error("Connection error:", err);
      setIsServerOnline(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000); 
    return () => clearInterval(interval);
  }, [isDirty]);

  const handleConfigChange = (setter, value) => {
    setter(value);
    setIsDirty(true);
  };

  // Відправка налаштувань
  const sendConfig = async () => {
    setIsSaving(true);
    try {
      const payload = {
        days_data: Number(days),
        epochs: Number(epochs)
      };
      
      await fetch(`${API_URL}/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      setIsDirty(false);
    } catch (err) {
      alert("Помилка збереження");
    } finally {
      setTimeout(() => setIsSaving(false), 1000);
    }
  };

  if (!status && !isServerOnline) return (
    <div className="min-h-screen bg-[#0f172a] text-white flex items-center justify-center gap-2">
        <AlertCircle className="text-red-500" /> Сервер офлайн...
    </div>
  );

  const result = status?.result || {};
  const isTraining = status?.is_training;
  const isUp = result.prediction_prob > 0.5;
  const hasData = result.timestamp !== null;

  return (
    <div className={styles.container}>
      {/* Глобальні стилі (замість окремого index.css) */}
      <style>{`
        body { margin: 0; font-family: sans-serif; background-color: #0f172a; color: white; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
      `}</style>

      <header className={styles.header}>
        <div className="flex items-center gap-3">
          <div className="p-3 bg-blue-500/10 rounded-xl border border-blue-500/20">
            <Brain className="text-blue-400" size={32} />
          </div>
          <div>
            <h1 className={styles.title}>Ethereum Price Predictor</h1>
          </div>
        </div>
      </header>

      <div className={styles.grid}>
        
        {/* КАРТКА ПРОГНОЗУ */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className={`${styles.card} lg:col-span-1 flex flex-col justify-between`}
        >
          <div className="absolute top-0 right-0 p-32 bg-blue-500/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none"></div>
          
          <div>
            <div className="flex justify-between items-start mb-6 relative z-10">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <Activity className="text-cyan-400" />
                Прогноз
              </h2>
              {isTraining && <RefreshCw className="animate-spin text-yellow-400" size={20} />}
            </div>
            
            <div className="flex flex-col items-center justify-center py-4 min-h-[200px] relative z-10">
               <AnimatePresence mode="wait">
                 {isTraining && !hasData ? (
                   <motion.div 
                     key="loader"
                     initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                     className="text-center"
                   >
                     <RefreshCw size={64} className="text-slate-600 animate-spin mb-4 mx-auto" />
                     <p className="text-slate-500">Первинний аналіз ринку...</p>
                   </motion.div>
                 ) : (
                   <motion.div 
                    key="result"
                    initial={{ scale: 0.8, opacity: 0 }} 
                    animate={{ scale: 1, opacity: 1 }}
                    className="text-center"
                   >
                     {/* Велика іконка */}
                     {isUp ? (
                       <TrendingUp size={80} className="text-green-500 drop-shadow-[0_0_15px_rgba(16,185,129,0.5)] mx-auto mb-4" />
                     ) : (
                       <TrendingDown size={80} className="text-red-500 drop-shadow-[0_0_15px_rgba(239,68,68,0.5)] mx-auto mb-4" />
                     )}
                     
                     <div className="text-5xl font-black text-white tracking-tighter mb-2">
                       {(result.prediction_prob * 100).toFixed(1)}%
                     </div>
                     <div className={`text-lg font-bold tracking-widest uppercase ${isUp ? 'text-green-400' : 'text-red-400'}`}>
                       {isUp ? "Зростання (LONG)" : "Падіння (SHORT)"}
                     </div>
                   </motion.div>
                 )}
               </AnimatePresence>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 mt-6 pt-6 border-t border-slate-700/50 relative z-10">
            <div>
              <span className={styles.statLabel}>Ціна зараз (USD)</span>
              <div className={styles.statValue}>
                {result.last_price?.toLocaleString('uk-UA', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </div>
            </div>
            
            <div className="relative group">
              <span className={styles.statLabel}>
                Точність (Test) 
                <HelpCircle size={14} className="text-slate-500 cursor-help" />
                <div className="absolute bottom-full left-0 mb-2 w-48 p-2 bg-slate-800 text-xs text-slate-300 rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 border border-slate-600">
                  Відсоток правильних прогнозів напрямку ціни на історичних даних.
                </div>
              </span>
              <div className={`text-2xl font-bold mt-1 ${result.accuracy > 0.6 ? 'text-green-400' : 'text-yellow-400'}`}>
                {(result.accuracy * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </motion.div>

        {/* ГРАФІК */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className={`${styles.card} lg:col-span-2 flex flex-col`}
        >
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <TrendingUp className="text-purple-400" />
              Графік ETH (Останній рік)
            </h2>
          </div>
          
          <div className="w-full h-[400px]">
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart 
                    data={chartData} 
                    margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} minTickGap={40} />
                  <YAxis stroke="#94a3b8" fontSize={12} domain={['auto', 'auto']} width={60} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#475569', color: '#fff' }}
                    labelFormatter={(label) => `Дата: ${label}`}
                    formatter={(value) => [`$${value.toLocaleString('uk-UA', {minimumFractionDigits: 2})}`, 'Ціна']}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="price" 
                    stroke="#8b5cf6" 
                    strokeWidth={2} 
                    fill="url(#colorPrice)" 
                    activeDot={{ r: 6 }} 
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
               <div className="h-full flex items-center justify-center text-slate-500">Завантаження даних...</div>
            )}
          </div>
        </motion.div>

        {/* НАЛАШТУВАННЯ */}
        <motion.div 
           initial={{ opacity: 0, y: 20 }}
           animate={{ opacity: 1, y: 0 }}
           transition={{ delay: 0.2 }}
           className={`${styles.card} lg:col-span-3`}
        >
           <h2 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
              <Settings className="text-slate-400" />
              Налаштування Нейромережі
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 items-end">
              <div>
                <label className={styles.label}>Глибина історії (Днів)</label>
                <input type="number" value={days} onChange={(e) => handleConfigChange(setDays, e.target.value)} className={styles.input} />
              </div>
              
              <div>
                <label className={styles.label}>Епохи навчання</label>
                <input type="number" value={epochs} onChange={(e) => handleConfigChange(setEpochs, e.target.value)} className={styles.input} />
              </div>

              <div className="hidden md:block"></div>

              <button 
                onClick={() => sendConfig()}
                disabled={isSaving} 
                className={`${styles.button} ${isSaving ? 'bg-slate-700' : 'bg-blue-600 hover:bg-blue-500 text-white'}`}
              >
                {isSaving ? "Збереження..." : isTraining ? "Зберегти (Застосується пізніше)" : "Застосувати"}
              </button>
            </div>
        </motion.div>

      </div>
    </div>
  );
}