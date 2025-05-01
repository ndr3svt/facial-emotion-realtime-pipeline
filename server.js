// server.js - Bun.js server for handling client connections and API requests

import { serve } from 'bun';
import { readFileSync, writeFileSync } from 'node:fs';

// Configuration
const FLASK_API_URL = 'http://localhost:3440/predict-simple'; // Use the simplified endpoint

// Helper function to save blob data for debugging
async function saveBlobToFile(blob, filename) {
  try {
    const buffer = await blob.arrayBuffer();
    writeFileSync(filename, Buffer.from(buffer));
    console.log(`Saved blob to ${filename}`);
  } catch (error) {
    console.error(`Failed to save blob: ${error.message}`);
  }
}

// Helper function to create a simple test image
function createTestImage(width, height) {
  // Create a canvas in memory
  const { createCanvas } = require('canvas');
  const canvas = createCanvas(width, height);
  const ctx = canvas.getContext('2d');
  
  // Draw a simple gradient
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, 'red');
  gradient.addColorStop(0.5, 'green');
  gradient.addColorStop(1, 'blue');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  
  // Draw a face-like shape
  ctx.fillStyle = '#FFF9E6';
  ctx.beginPath();
  ctx.arc(width/2, height/2, Math.min(width, height)/3, 0, Math.PI * 2);
  ctx.fill();
  
  // Draw eyes
  ctx.fillStyle = 'black';
  ctx.beginPath();
  ctx.arc(width/2 - 20, height/2 - 10, 10, 0, Math.PI * 2);
  ctx.arc(width/2 + 20, height/2 - 10, 10, 0, Math.PI * 2);
  ctx.fill();
  
  // Draw smile
  ctx.beginPath();
  ctx.arc(width/2, height/2 + 10, 30, 0, Math.PI);
  ctx.stroke();
  
  // Convert to JPEG buffer
  return canvas.toBuffer('image/jpeg');
}

// Create a simple HTTP server
const server = serve({
  port: 3000,
  async fetch(req) {
    const url = new URL(req.url);
    
    // Serve static files
    if (url.pathname === '/') {
      return new Response(readFileSync('./index.html'), {
        headers: { 'Content-Type': 'text/html' },
      });
    }
    
    // Test endpoint - creates and sends a test image
    if (url.pathname === '/api/test-image') {
      try {
        // Create a test image
        const imageBuffer = createTestImage(640, 480);
        const base64Image = imageBuffer.toString('base64');
        
        console.log(`Created test image (${base64Image.length} base64 chars)`);
        
        // Send to Flask API
        const apiResponse = await fetch(FLASK_API_URL, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            image_base64: base64Image
          }),
        });
        
        if (!apiResponse.ok) {
          const errorText = await apiResponse.text();
          return new Response(errorText, { status: apiResponse.status });
        }
        
        const result = await apiResponse.json();
        return Response.json(result);
      } catch (error) {
        console.error(`Error in test image: ${error.message}`);
        return Response.json({ error: error.message }, { status: 500 });
      }
    }
    
    // API endpoint to proxy camera frames to Flask
    if (url.pathname === '/api/emotion' && req.method === 'POST') {
      try {
        console.log("Received request to /api/emotion");
        
        // Get the image data from the request as text
        // For JSON requests with base64 encoded data
        if (req.headers.get('Content-Type')?.includes('application/json')) {
          const jsonData = await req.json();
          
          if (!jsonData.image_base64) {
            return Response.json({ error: 'No image_base64 field in request' }, { status: 400 });
          }
          
          const base64Image = jsonData.image_base64;
          console.log(`Received JSON with base64 image (${base64Image.length} chars)`);
          
          // Save a sample of the base64 data for debugging
          writeFileSync('debug_base64_sample.txt', base64Image.substring(0, 1000) + '...');
          
          // Forward the request directly to Flask API
          console.log(`Sending base64 image to Flask API (${base64Image.length} chars)`);
          const apiResponse = await fetch(FLASK_API_URL, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image_base64: base64Image
            }),
          });
          
          // Handle response
          console.log(`Response status: ${apiResponse.status}`);
          
          if (!apiResponse.ok) {
            const errorText = await apiResponse.text();
            console.error(`Flask API error: ${errorText}`);
            return Response.json({ 
              error: `Flask API returned ${apiResponse.status}`,
              details: errorText
            }, { status: apiResponse.status });
          }
          
          // Parse response
          const responseText = await apiResponse.text();
          try {
            const result = JSON.parse(responseText);
            return Response.json(result);
          } catch (e) {
            console.error(`Failed to parse response as JSON: ${e.message}`);
            return Response.json({ 
              error: 'Invalid JSON response from server',
              rawResponse: responseText.substring(0, 500)
            }, { status: 500 });
          }
        } 
        // For multipart form data or raw binary data
        else { 
          const imageBlob = await req.blob();
          console.log(`Image blob size: ${imageBlob.size}, type: ${imageBlob.type}`);
          
          // Save the blob to a file for debugging
          await saveBlobToFile(imageBlob, 'debug_request_blob.bin');
          
          // Skip if the blob is empty or too small
          if (!imageBlob || imageBlob.size < 100) {
            return Response.json({ error: 'Empty or invalid image data' }, { status: 400 });
          }
          
          // Convert the raw blob to base64
          const arrayBuffer = await imageBlob.arrayBuffer();
          const buffer = Buffer.from(arrayBuffer);
          const base64Image = buffer.toString('base64');
          
          // Forward the request to Flask API using JSON with base64 data
          console.log(`Sending base64 image to Flask API (${base64Image.length} chars)`);
          const apiResponse = await fetch(FLASK_API_URL, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image_base64: base64Image
            }),
          });
          
          // Handle response
          console.log(`Response status: ${apiResponse.status}`);
          
          if (!apiResponse.ok) {
            const errorText = await apiResponse.text();
            console.error(`Flask API error: ${errorText}`);
            return Response.json({ 
              error: `Flask API returned ${apiResponse.status}`,
              details: errorText
            }, { status: apiResponse.status });
          }
          
          // Parse response
          const responseText = await apiResponse.text();
          try {
            const result = JSON.parse(responseText);
            return Response.json(result);
          } catch (e) {
            console.error(`Failed to parse response as JSON: ${e.message}`);
            return Response.json({ 
              error: 'Invalid JSON response from server',
              rawResponse: responseText.substring(0, 500)
            }, { status: 500 });
          }
        }
      } catch (error) {
        console.error(`Error in emotion detection: ${error.message}`);
        console.error(error.stack);
        return Response.json({ 
          error: 'Failed to process the image',
          details: error.message
        }, { status: 500 });
      }
    }
    
    // Debug endpoint for image verification
    if (url.pathname === '/api/debug-image' && req.method === 'POST') {
      try {
        console.log("Received request to /api/debug-image");
        
        // For JSON requests with base64 encoded data
        if (req.headers.get('Content-Type')?.includes('application/json')) {
          const jsonData = await req.json();
          
          if (!jsonData.image_base64) {
            return Response.json({ error: 'No image_base64 field in request' }, { status: 400 });
          }
          
          const base64Image = jsonData.image_base64;
          console.log(`Received JSON with base64 image for debug (${base64Image.length} chars)`);
          
          // Forward to Flask debug endpoint
          const apiResponse = await fetch(`http://localhost:3440/debug-image`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image_base64: base64Image
            }),
          });
          
          if (!apiResponse.ok) {
            const errorText = await apiResponse.text();
            console.error(`Flask debug API error: ${errorText}`);
            return Response.json({ 
              error: `Flask API returned ${apiResponse.status}`,
              details: errorText
            }, { status: apiResponse.status });
          }
          
          // Pass through the response
          const responseData = await apiResponse.json();
          return Response.json(responseData);
        } 
        else { 
          // Handle non-JSON requests (like form data)
          const imageBlob = await req.blob();
          console.log(`Debug image blob size: ${imageBlob.size}, type: ${imageBlob.type}`);
          
          // Convert blob to base64
          const arrayBuffer = await imageBlob.arrayBuffer();
          const buffer = Buffer.from(arrayBuffer);
          const base64Image = buffer.toString('base64');
          
          // Forward to Flask debug endpoint
          const apiResponse = await fetch(`http://localhost:3440/debug-image`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image_base64: base64Image
            }),
          });
          
          if (!apiResponse.ok) {
            const errorText = await apiResponse.text();
            console.error(`Flask debug API error: ${errorText}`);
            return Response.json({ 
              error: `Flask API returned ${apiResponse.status}`,
              details: errorText
            }, { status: apiResponse.status });
          }
          
          // Pass through the response
          const responseData = await apiResponse.json();
          return Response.json(responseData);
        }
      } catch (error) {
        console.error(`Error in debug endpoint: ${error.message}`);
        return Response.json({ 
          error: 'Failed to process the debug image',
          details: error.message
        }, { status: 500 });
      }
    }
    
    // Proxy endpoint for debug images
    if (url.pathname.startsWith('/debug_') && url.pathname.endsWith('.jpg')) {
      try {
        const filename = url.pathname.substring(1); // Remove leading slash
        const imageResponse = await fetch(`http://localhost:3440${url.pathname}`);
        
        if (!imageResponse.ok) {
          return new Response('Image not found', { status: 404 });
        }
        
        const imageBlob = await imageResponse.blob();
        return new Response(imageBlob, {
          headers: { 'Content-Type': 'image/jpeg' }
        });
      } catch (error) {
        console.error(`Error fetching debug image: ${error.message}`);
        return new Response('Error fetching image', { status: 500 });
      }
    }
    
    // Health check endpoint
    if (url.pathname === '/api/health') {
      try {
        const apiResponse = await fetch(`http://localhost:3440/health`);
        const result = await apiResponse.json();
        return Response.json(result);
      } catch (error) {
        return Response.json({ 
          error: 'Flask server not available',
          details: error.message
        }, { status: 503 });
      }
    }
    
    // Handle 404 for other routes
    return new Response('Not Found', { status: 404 });
  },
});

console.log(`Server running at http://localhost:${server.port}`);
console.log(`Using Flask API at: ${FLASK_API_URL}`);