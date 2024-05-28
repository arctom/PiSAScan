import React from 'react';
import { Helmet } from 'react-helmet';
import './index.css';

const PiSAScan = () => {
  return (
    <>
      <Helmet>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta name="theme-color" content="#FFFFFF" />
        <link rel="icon" type="image/png" href="public/image/electrolit.png" />
        <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Varela+Round" />
        <link rel="stylesheet" href="index.css" />
        <title>PiSA Scan</title>
      </Helmet>
      <div className="container">
        <div className="navbar">
          <img src="image/electrolit.png" alt="Favicon" style={{ height: '100px' }} />
          <h1>PiSA Scan</h1>
        </div>
        <div id="dropCV">
          <h2>Drop CV</h2>
          <label>
            <input type="file" accept=".pdf,.doc,.docx" />
            <span className="label_top">Upload</span>
          </label>
        </div>
      </div>
      <footer>
        <p>&copy; 2024 Grupo PiSA. All rights reserved.</p>
      </footer>
    </>
  );
};

export default PiSAScan;
