import {Route, Routes } from 'react-router-dom';
import About from './pages/About/about';
import Trading from './pages/Trading/trading';

function App() {
  return (
    <>
      <Routes>
        <Route exact path="/" element={<Trading/>}/>
        <Route exact path="/about" element={<About/>}/>
      </Routes>
    </>
  )
}

export default App
