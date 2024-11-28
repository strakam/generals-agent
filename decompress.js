const axios = require('axios');
const LZString = require('lz-string');
const { Buffer } = require('buffer');

async function decompressData(url) {
    try {
        // Fetch the compressed data from the provided URL
        const response = await axios.get(url, { responseType: 'arraybuffer' });

        // Convert the downloaded data (arraybuffer) to a Uint8Array
        const uint8Array = new Uint8Array(response.data);

        // Decompress using LZString.decompressFromUint8Array
        const decompressedData = LZString.decompressFromUint8Array(uint8Array);

        // Print the decompressed data so Python can capture it
        console.log(decompressedData);
    } catch (error) {
        console.error("Error downloading or decompressing data:", error);
        process.exit(1);
    }
}

// Get URL from command line arguments
const url = process.argv[2];
decompressData(url);
